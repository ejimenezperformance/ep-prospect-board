"""Ways to turn several scout reports (+ measurables) into one board score.

a) one_scout  : the first report only (what you have with one look)
b) average    : plain average of all FV grades (the usual consensus)
c) weighted   : each scout's grade corrected for that scout's learned bias and maturity
                lean, then weighted by how consistent the scout has been (1 / residual variance)
d) full       : weighted score + measurables + age in a logistic model trained on past classes
e) tools_only : measurables + age only (no scouts), for context

Scout parameters are learned ONLY from training classes, by comparing each scout with
the other scouts who saw the same player. No outcomes are needed for that step, so a
club can update it the same summer, years before anyone reaches Double-A.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MEAS = ["bat_speed", "exit_velo", "sprint"]


def age_c(age):
    return (np.asarray(age) - 16.8) / 0.46


def learn_scouts(reports: pd.DataFrame, prospects: pd.DataFrame) -> pd.DataFrame:
    r = reports.merge(prospects[["pid", "age"]], on="pid")
    r["g"] = (r.fv - 45) / 7.5
    tot = r.groupby("pid").g.agg(["sum", "count"])
    r = r.join(tot, on="pid")
    r = r[r["count"] >= 2].copy()
    r["others"] = (r["sum"] - r.g) / (r["count"] - 1)
    r["resid"] = r.g - r.others
    r["ac"] = age_c(r.age)
    out, clean = [], {}
    for s, d in r.groupby("scout"):
        X = np.column_stack([np.ones(len(d)), d.ac])
        beta, *_ = np.linalg.lstsq(X, d.resid, rcond=None)
        res = d.resid - X @ beta
        clean[s] = float(np.var(res, ddof=2))
        out.append({"scout": s, "bias_hat": beta[0], "maturity_hat": beta[1],
                    "resid_var": clean[s], "n_reports": len(d)})
    sp = pd.DataFrame(out).set_index("scout")

    # A scout's residual against the others mixes his own noise with theirs:
    #   Var(resid_s) = noise_s + mean over his reports of [ sum(noise_o for o in others) / (k-1)^2 ]
    # Solve for each scout's own noise by iterating (method of moments).
    members = r.groupby("pid").scout.apply(list)
    r = r.join(members.rename("crew"), on="pid")
    noise = {s: v / 2 for s, v in clean.items()}
    for _ in range(25):
        new = {}
        for s, d in r.groupby("scout"):
            c = d.apply(lambda x: sum(noise[o] for o in x.crew if o != s) / (x["count"] - 1) ** 2, axis=1).mean()
            new[s] = max(clean[s] - c, 0.02)
        noise = new
    sp["noise_hat"] = pd.Series(noise)
    sp["weight"] = 1 / sp.noise_hat
    sp["weight"] /= sp.weight.mean()
    return sp.reset_index()


def weighted_score(reports, prospects, scout_params):
    r = reports.merge(prospects[["pid", "age"]], on="pid").merge(scout_params, on="scout", how="left")
    r["bias_hat"] = r.bias_hat.fillna(0)
    r["maturity_hat"] = r.maturity_hat.fillna(0)
    r["weight"] = r.weight.fillna(1)
    r["corr"] = (r.fv - 45) / 7.5 - r.bias_hat - r.maturity_hat * age_c(r.age)
    g = r.groupby("pid").apply(lambda d: pd.Series({
        "weighted": np.average(d["corr"], weights=d.weight),
        "weighted_se": float(np.sqrt(1 / d.weight.sum())),
    }), include_groups=False)
    return prospects.join(g, on="pid")


def first_report(reports, prospects):
    f = reports.groupby("pid").fv.first().rename("one_scout")
    return prospects.join(f, on="pid")


def _model(C=1.0):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=1000))


def features(df, kind):
    ac = age_c(df.age)
    if kind == "full":
        return np.column_stack([df.weighted] + [df[m] for m in MEAS] + [ac])
    if kind == "tools_only":
        return np.column_stack([df[m] for m in MEAS] + [ac])
    raise ValueError(kind)


def fit_predict(train, test, kind, n_boot=0, rng=None):
    m = _model().fit(features(train, kind), train.reached_aa)
    p = m.predict_proba(features(test, kind))[:, 1]
    if not n_boot:
        return p, None
    rng = rng or np.random.default_rng(0)
    boots = np.empty((n_boot, len(test)))
    for b in range(n_boot):
        idx = rng.integers(0, len(train), len(train))
        tb = train.iloc[idx]
        boots[b] = _model().fit(features(tb, kind), tb.reached_aa).predict_proba(features(test, kind))[:, 1]
    return p, boots


def build_scores(prospects, reports, train_classes, test_classes):
    tr = prospects[prospects["class"].isin(train_classes)]
    te = prospects[prospects["class"].isin(test_classes)]
    sp = learn_scouts(reports[reports["class"].isin(train_classes)], tr)
    tr = weighted_score(reports, tr, sp)
    te = weighted_score(reports, te, sp)
    te = first_report(reports, te)
    te["average"] = te.consensus_fv
    te["full"], _ = fit_predict(tr, te, "full")
    te["tools_only"], _ = fit_predict(tr, te, "tools_only")
    return tr, te, sp
