"""Charts in EP style. Usage: python src/charts.py results figures [en|es]"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT, FIG = Path(sys.argv[1]), Path(sys.argv[2])
LANG = sys.argv[3] if len(sys.argv) > 3 else "en"
FIG.mkdir(parents=True, exist_ok=True)
NAVY, GOLD, BLUE, MUTED, GRID = "#0B1B33", "#B8862A", "#2A62A8", "#6B7280", "#E5E1D6"
plt.rcParams.update({"font.family": ["DejaVu Sans"], "axes.edgecolor": MUTED, "text.color": NAVY,
                     "xtick.color": NAVY, "ytick.color": NAVY, "axes.labelcolor": NAVY, "font.size": 13})

T = {
    "en": {
        "strat": {"Market (average FV)": "Market\n(average FV)", "Weighted scouts": "Weighted\nscouts",
                  "Board (weighted scouts + measurables)": "Board\n(scouts + measurables)"},
        "hits": "Expected future Double-A players per class",
        "budget_title": "Same $6.5M pool, same 30 signings",
        "vs": "vs market",
        "meth": {"one_scout": "One scout", "average": "Average of scouts", "weighted": "Weighted scouts",
                 "full": "Weighted scouts + measurables", "tools_only": "Measurables only"},
        "auc": "AUC for reaching Double-A (held-out classes, 95% CI)",
        "noise": "True scout noise (higher = less accurate)",
        "weight": "Weight the board learned",
        "learn_title": "The board learns who to trust (no outcomes needed)",
        "rho": "Spearman ρ = {r:.2f}",
    },
    "es": {
        "strat": {"Market (average FV)": "Mercado\n(FV promedio)", "Weighted scouts": "Scouts\nponderados",
                  "Board (weighted scouts + measurables)": "Board\n(scouts + medibles)"},
        "hits": "Futuros jugadores de Doble-A esperados por clase",
        "budget_title": "Mismo pool de $6.5M, mismas 30 firmas",
        "vs": "vs. mercado",
        "meth": {"one_scout": "Un scout", "average": "Promedio de scouts", "weighted": "Scouts ponderados",
                 "full": "Scouts ponderados + medibles", "tools_only": "Solo medibles"},
        "auc": "AUC para llegar a Doble-A (clases fuera de muestra, IC 95%)",
        "noise": "Ruido real del scout (más alto = menos preciso)",
        "weight": "Peso que aprendió el board",
        "learn_title": "El board aprende en quién confiar (sin esperar resultados)",
        "rho": "ρ de Spearman = {r:.2f}",
    },
}[LANG]
suf = "" if LANG == "en" else "_es"


def clean(ax, axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


# 1. Budget test (headline)
per = pd.read_csv(OUT / "budget_by_class.csv")
summ = pd.read_csv(OUT / "budget_summary.csv").set_index("strategy")
order = ["Market (average FV)", "Weighted scouts", "Board (weighted scouts + measurables)"]
fig, ax = plt.subplots(figsize=(10, 5.6))
cols = [MUTED, BLUE, GOLD]
for i, s in enumerate(order):
    v = per[per.strategy == s].expected_hits
    m = v.mean()
    se = v.std(ddof=1) / np.sqrt(len(v))
    ax.bar(i, m, 0.6, color=cols[i])
    ax.errorbar(i, m, yerr=1.96 * se, color=NAVY, capsize=6, lw=1.5)
    txt = f"{m:.1f}"
    if s != order[0]:
        g = summ.loc[s]
        txt += f"\n{g.gain_vs_market:+.0%} {T['vs']}\n[{g.ci_lo:+.0%}, {g.ci_hi:+.0%}]"
    ax.text(i, m + 1.96 * se + 0.3, txt, ha="center", va="bottom", fontsize=12, fontweight="bold")
ax.set_xticks(range(3), [T["strat"][s] for s in order])
ax.set_ylabel(T["hits"])
ax.set_ylim(0, per.expected_hits.max() * 1.35)
ax.set_title(T["budget_title"], loc="left", fontweight="bold")
clean(ax)
fig.tight_layout(); fig.savefig(FIG / f"budget{suf}.png", dpi=200); plt.close(fig)

# 2. AUC by method
auc = pd.read_csv(OUT / "auc_by_method.csv").sort_values("auc")
fig, ax = plt.subplots(figsize=(10, 4.6))
for i, r in enumerate(auc.itertuples()):
    c = GOLD if r.method == "full" else NAVY
    ax.plot([r.ci_lo, r.ci_hi], [i, i], color="#C9CFD8", lw=8, solid_capstyle="round")
    ax.scatter(r.auc, i, s=140, color=c, zorder=3, edgecolor="white", lw=1.5)
    ax.text(r.ci_hi + 0.004, i, f"{r.auc:.3f}", va="center", fontsize=12)
ax.set_yticks(range(len(auc)), [T["meth"][m] for m in auc.method])
ax.set_xlabel(T["auc"])
ax.set_xlim(0.65, 0.80)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(FIG / f"auc{suf}.png", dpi=200); plt.close(fig)

# 3. Scout learning
s = pd.read_csv(OUT / "scouts_true_vs_learned.csv")
from scipy.stats import spearmanr  # noqa: E402
rho = spearmanr(s.true_noise, s.weight)[0]
fig, ax = plt.subplots(figsize=(9, 5.4))
ax.scatter(s.true_noise, s.weight, s=110, color=BLUE, edgecolor="white", lw=1.5, zorder=3)
for r in s.itertuples():
    ax.annotate(r.scout, (r.true_noise, r.weight), xytext=(6, 4), textcoords="offset points", fontsize=10, color=MUTED)
ax.set_xlabel(T["noise"]); ax.set_ylabel(T["weight"])
ax.set_title(T["learn_title"], loc="left", fontweight="bold")
ax.text(0.98, 0.95, T["rho"].format(r=rho), transform=ax.transAxes, ha="right", fontsize=12)
clean(ax, "both")
fig.tight_layout(); fig.savefig(FIG / f"scout_learning{suf}.png", dpi=200); plt.close(fig)
print("ok", LANG)
