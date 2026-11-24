"""Run the full study on the simulated world and write results/.

Usage:  python src/evaluate.py results
Rule fixed before running (same as the EP international pipeline):
  ranking  -> CONFIRMED if AUC gain >= 0.02 and its paired bootstrap 95% CI is above zero
  budget   -> CONFIRMED if expected hits rise >= 10% and the bootstrap 95% CI (over classes) is above zero
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from simulate import World, simulate  # noqa: E402
from board import build_scores, fit_predict, weighted_score, learn_scouts  # noqa: E402

METHODS = ["one_scout", "average", "weighted", "full", "tools_only"]
LABELS = {"one_scout": "One scout", "average": "Average of scouts", "weighted": "Weighted scouts",
          "full": "Weighted scouts + measurables", "tools_only": "Measurables only"}
PAIRS = [("average", "one_scout"), ("weighted", "average"), ("full", "weighted")]
ROSTER_CAP = 30


def paired_auc(y, a, b, n_boot=2000, seed=1):
    rng = np.random.default_rng(seed)
    y = np.asarray(y); a = np.asarray(a); b = np.asarray(b)
    base = roc_auc_score(y, a) - roc_auc_score(y, b)
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        d.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return base, lo, hi


def label(gain, lo, threshold=0.02):
    return "CONFIRMED" if gain >= threshold and lo > 0 else "NOT CONFIRMED"


def _greedy(d, key, pool):
    d = d.assign(key=key).sort_values("key", ascending=False)
    spent, picks = 0.0, []
    for pid, b in zip(d.pid, d.bonus_ask):
        if len(picks) >= ROSTER_CAP:
            break
        if spent + b <= pool:
            picks.append(pid); spent += b
    return d[d.pid.isin(picks)]


def pick_class(df, score_col, pool):
    """Sign exactly ROSTER_CAP players and maximize the total score within the pool.

    Ranks on score - lambda * bonus and raises lambda until the signings fit the pool.
    Every strategy uses the same optimizer and roster size; only the score differs.
    """
    sc = df[score_col]
    sc = (sc - sc.mean()) / sc.std()
    hi = 1e-7
    while _greedy(df, sc - hi * df.bonus_ask, np.inf).bonus_ask.sum() > pool:
        hi *= 2
    lo = 0.0
    for _ in range(40):
        lam = (lo + hi) / 2
        if _greedy(df, sc - lam * df.bonus_ask, np.inf).bonus_ask.sum() <= pool:
            hi = lam
        else:
            lo = lam
    return _greedy(df, sc - hi * df.bonus_ask, np.inf)


def budget_test(te, pool, n_boot=2000, seed=2):
    strategies = {"Market (average FV)": "average",
                  "Weighted scouts": "weighted",
                  "Board (weighted scouts + measurables)": "full"}
    rows = []
    for c, d in te.groupby("class"):
        for name, col in strategies.items():
            p = pick_class(d, col, pool)
            rows.append({"class": c, "strategy": name, "signed": len(p), "spent": p.bonus_ask.sum(),
                         "expected_hits": p.p_true.sum(), "actual_hits": int(p.reached_aa.sum())})
    per = pd.DataFrame(rows)
    piv = per.pivot(index="class", columns="strategy", values="expected_hits")
    base = "Market (average FV)"
    rng = np.random.default_rng(seed)
    summ = []
    for s in strategies:
        g = piv[s].sum() / piv[base].sum() - 1
        bs = []
        for _ in range(n_boot):
            i = rng.integers(0, len(piv), len(piv))
            bs.append(piv[s].iloc[i].sum() / piv[base].iloc[i].sum() - 1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        agg = per[per.strategy == s]
        summ.append({"strategy": s, "signed_per_class": agg.signed.mean(), "spent_per_class": agg.spent.mean(),
                     "expected_hits_per_class": agg.expected_hits.mean(), "actual_hits_per_class": agg.actual_hits.mean(),
                     "gain_vs_market": g, "ci_lo": lo, "ci_hi": hi,
                     "label": "" if s == base else label(g, lo, 0.10)})
    return per, pd.DataFrame(summ)


def run(world: World, n_train=15):
    pros, reps, scouts = simulate(world)
    classes = sorted(pros["class"].unique())
    tr_c, te_c = classes[:n_train], classes[n_train:]
    tr, te, sp = build_scores(pros, reps, tr_c, te_c)
    return pros, reps, scouts, tr, te, sp


def main(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    w = World()
    pros, reps, scouts, tr, te, sp = run(w)
    y = te.reached_aa

    auc = pd.DataFrame([{"method": m, "label": LABELS[m], "auc": roc_auc_score(y, te[m])} for m in METHODS])
    rng = np.random.default_rng(3)
    ci = []
    for m in METHODS:
        bs = []
        for _ in range(2000):
            i = rng.integers(0, len(te), len(te))
            bs.append(roc_auc_score(y.iloc[i], te[m].iloc[i]))
        ci.append(np.percentile(bs, [2.5, 97.5]))
    auc["ci_lo"], auc["ci_hi"] = [c[0] for c in ci], [c[1] for c in ci]
    auc.to_csv(out / "auc_by_method.csv", index=False)

    gains = []
    for a, b in PAIRS:
        g, lo, hi = paired_auc(y, te[a], te[b])
        gains.append({"comparison": f"{LABELS[a]} vs {LABELS[b]}", "gain": g, "ci_lo": lo, "ci_hi": hi,
                      "label": label(g, lo)})
    gains = pd.DataFrame(gains)
    gains.to_csv(out / "auc_gains.csv", index=False)

    # Did the board learn which scouts are good? (true noise vs learned weight)
    s = scouts.merge(sp, on="scout")
    rho, pval = spearmanr(s.true_noise, s.weight)
    rho_b, _ = spearmanr(s.true_bias, s.bias_hat)
    rho_m, _ = spearmanr(s.true_maturity, s.maturity_hat)
    s.to_csv(out / "scouts_true_vs_learned.csv", index=False)

    per, budget = budget_test(te, w.pool_per_class)
    per.to_csv(out / "budget_by_class.csv", index=False)
    budget.to_csv(out / "budget_summary.csv", index=False)

    # Example board for the last class, with an 80% interval from refitting on bootstrap samples
    last = te[te["class"] == te["class"].max()].copy()
    p, boots = fit_predict(tr, last, "full", n_boot=300, rng=np.random.default_rng(4))
    last["p_board"] = p
    last["p_lo"], last["p_hi"] = np.percentile(boots, 10, axis=0), np.percentile(boots, 90, axis=0)
    last["value_per_100k"] = last.p_board / (last.bonus_ask / 100_000)
    mkt_rank = last.average.rank(ascending=False, method="min")
    last["market_rank"] = mkt_rank.astype(int)
    last["board_rank"] = last.p_board.rank(ascending=False, method="min").astype(int)
    cols = ["pid", "region", "age", "n_reports", "average", "weighted", "bat_speed", "exit_velo", "sprint",
            "bonus_ask", "p_board", "p_lo", "p_hi", "value_per_100k", "market_rank", "board_rank", "p_true", "reached_aa"]
    last.sort_values("p_board", ascending=False)[cols].head(25).to_csv(out / "example_board_top25.csv", index=False)

    summary = {
        "world": {"classes": w.n_classes, "per_class": w.n_per_class, "train_classes": 15, "test_classes": int(w.n_classes - 15),
                  "scouts": int(len(scouts)), "reports": int(len(reps)), "base_rate": float(pros.reached_aa.mean())},
        "auc": auc.set_index("method").auc.round(3).to_dict(),
        "gains": gains.round(3).to_dict("records"),
        "scout_learning": {"spearman_noise_vs_weight": round(float(rho), 2), "p": float(pval),
                           "spearman_bias": round(float(rho_b), 2), "spearman_maturity": round(float(rho_m), 2)},
        "budget": budget.round(3).to_dict("records"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
