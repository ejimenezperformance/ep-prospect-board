"""Known-answer checks in both directions, run before trusting the main results.

If the code works, it must find a gain when one is built into the world, and must
NOT find one when the world has none. Run:  python tests/test_known_answer.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from simulate import World  # noqa: E402
from evaluate import run, paired_auc  # noqa: E402

SMALL = dict(n_classes=16, n_per_class=400)


def gain(world, a, b):
    *_, te, _ = run(world, n_train=8)
    return paired_auc(te.reached_aa, te[a], te[b], n_boot=500)


def check(name, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


results = []

# 1) Scouts all identical (same noise, no bias, no maturity lean): weighting cannot help.
g, lo, hi = gain(World(scout_noise=(1.0, 1.0), scout_bias=(0, 0), scout_maturity=(0, 0), seed=11, **SMALL),
                 "weighted", "average")
results.append(check("Identical scouts -> no weighting gain", abs(g) < 0.02 and lo < 0 < hi,
                     f"gain {g:+.3f} [{lo:+.3f}, {hi:+.3f}]"))

# 2) Scouts very different (some sharp, some close to guessing, strong biases): weighting must help.
g, lo, hi = gain(World(scout_noise=(0.3, 3.0), scout_bias=(-1.0, 1.0), scout_maturity=(0, 1.0), seed=12, **SMALL),
                 "weighted", "average")
results.append(check("Very different scouts -> weighting gain", g >= 0.02 and lo > 0,
                     f"gain {g:+.3f} [{lo:+.3f}, {hi:+.3f}]"))

# 3) Measurables are pure noise (no talent signal, no maturity): adding them cannot help.
g, lo, hi = gain(World(measurable_signal={"bat_speed": 0.0, "exit_velo": 0.0, "sprint": 0.0},
                       maturity_on_measurables=0.0, seed=13, **SMALL), "full", "weighted")
results.append(check("Noise measurables -> no gain", g < 0.02 and lo < 0.005,
                     f"gain {g:+.3f} [{lo:+.3f}, {hi:+.3f}]"))

# 4) Measurables strongly tied to talent: adding them must help.
g, lo, hi = gain(World(measurable_signal={"bat_speed": 0.85, "exit_velo": 0.8, "sprint": 0.6}, seed=14, **SMALL),
                 "full", "weighted")
results.append(check("Strong measurables -> gain", g >= 0.02 and lo > 0,
                     f"gain {g:+.3f} [{lo:+.3f}, {hi:+.3f}]"))

print(f"\n{sum(results)}/{len(results)} known-answer checks passed")
sys.exit(0 if all(results) else 1)
