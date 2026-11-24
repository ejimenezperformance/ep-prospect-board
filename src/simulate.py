"""Synthetic international signing classes.

Everything here is SIMULATED. Real scouting grades and bonuses are private, so the
world is built with known truth: each prospect has a latent talent T, and the
outcome (reaching Double-A) depends on T only. Scouts, measurables and the market
see T through noise and bias. The point is to test how a board should combine them.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

REGIONS = ["DR", "VEN", "OTHER"]


@dataclass
class World:
    n_classes: int = 20
    n_per_class: int = 400
    scouts_per_region: int = 5
    reports_min: int = 2
    reports_max: int = 4
    # scout noise SD (in talent units) is drawn uniformly from this range
    scout_noise: tuple = (0.45, 1.5)
    # scout bias (grade shift, talent units)
    scout_bias: tuple = (-0.4, 0.4)
    # how much a scout over-rewards physical maturity (older = looks better)
    scout_maturity: tuple = (0.0, 0.5)
    # measurable signal: loading of each measurable on true talent
    measurable_signal: dict = field(default_factory=lambda: {"bat_speed": 0.55, "exit_velo": 0.45, "sprint": 0.30})
    maturity_on_measurables: float = 0.35
    outcome_intercept: float = -1.75
    outcome_slope: float = 1.35
    pool_per_class: float = 6_500_000
    seed: int = 7


def round5(x):
    return np.clip(np.round(x / 5) * 5, 20, 80)


def simulate(w: World) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return prospects, scout reports and the scout roster (with true parameters)."""
    rng = np.random.default_rng(w.seed)

    scouts = []
    for r in REGIONS:
        for k in range(w.scouts_per_region):
            scouts.append({
                "scout": f"{r}-{k + 1}",
                "region": r,
                "true_noise": rng.uniform(*w.scout_noise),
                "true_bias": rng.uniform(*w.scout_bias),
                "true_maturity": rng.uniform(*w.scout_maturity),
            })
    scouts = pd.DataFrame(scouts)

    rows, reps = [], []
    pid = 0
    for c in range(1, w.n_classes + 1):
        region = rng.choice(REGIONS, size=w.n_per_class, p=[0.5, 0.35, 0.15])
        talent = rng.normal(0, 1, w.n_per_class)
        age = rng.uniform(16.0, 17.6, w.n_per_class)
        age_c = (age - 16.8) / 0.46  # standardized
        meas = {}
        for m, load in w.measurable_signal.items():
            noise_sd = np.sqrt(max(1 - load ** 2, 1e-6))
            meas[m] = load * talent + w.maturity_on_measurables * age_c + rng.normal(0, noise_sd, w.n_per_class)
        p = 1 / (1 + np.exp(-(w.outcome_intercept + w.outcome_slope * talent)))
        reached = rng.random(w.n_per_class) < p

        for i in range(w.n_per_class):
            pid += 1
            pool = scouts[scouts.region == region[i]]
            k = rng.integers(w.reports_min, w.reports_max + 1)
            seen = pool.sample(k, random_state=int(rng.integers(1e9)))
            grades = []
            for _, s in seen.iterrows():
                signal = talent[i] + s.true_bias + s.true_maturity * age_c[i] + rng.normal(0, s.true_noise)
                fv = float(round5(45 + 7.5 * signal))
                grades.append(fv)
                reps.append({"pid": pid, "class": c, "scout": s.scout, "fv": fv})
            rows.append({
                "pid": pid, "class": c, "region": region[i], "age": round(age[i], 2),
                "bat_speed": meas["bat_speed"][i], "exit_velo": meas["exit_velo"][i], "sprint": meas["sprint"][i],
                "talent": talent[i], "p_true": p[i], "reached_aa": bool(reached[i]),
                "consensus_fv": float(np.mean(grades)), "n_reports": k,
            })
    prospects = pd.DataFrame(rows)
    reports = pd.DataFrame(reps)

    # Market price: the asking bonus follows the public consensus (average FV + tools), not truth.
    # Calibrated to a typical class: median ask ~$40k, top asks ~$1.5-3M.
    z = (prospects.consensus_fv - 45) / 7.5 + 0.25 * prospects.bat_speed
    z = (z - z.mean()) / z.std()
    bonus = np.exp(10.6 + 1.55 * z + rng.normal(0, 0.35, len(prospects)))
    prospects["bonus_ask"] = np.clip(np.round(bonus, -3), 10_000, 3_500_000)
    return prospects, reports, scouts
