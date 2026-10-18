"""Hero 4 figure: who will hear — private vs public final answers after debate (analysis/lit_debate.py requery table),
with the internal read-out panel added when analysis/lit_hollow.py tables exist.
Usage: python figures/fig_lit_audience.py Q9 [L31 G4]
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "figures"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from common import *  # noqa
import style
from style import plt
from stats import fast_cluster_mean_ci

style.apply()
MNAME = {"Q9": "Qwen3.5-9B", "L31": "Llama-3.1-8B", "G4": "Gemma-4-E4B"}
DOMS = [("knowledge_reasoning", "Knowledge"), ("safety_refusal", "Safety"), ("values_moral_judgment", "Values")]
PRED = [("neg_margin", "unprinted answer margin (participant)"), ("neg_margin_observer", "same margin, outside reader"),
        ("tfidf", "text of the final response"), ("public_hist", "visible history (changes, hedging, agreement)"),
        ("neg_hedge", "hedging words")]


def ci(x, c, scale=100):
    e, lo, hi, *_ = fast_cluster_mean_ci(np.asarray(x, float), np.asarray(c))
    return scale * e, scale * lo, scale * hi


def hdot(ax, y, e, lo, hi, c, m="o", filled=True, label=None, s=32):
    ax.plot([lo, hi], [y, y], color=c, lw=1.3, solid_capstyle="round", zorder=2)
    ax.scatter([e], [y], s=s, marker=m, facecolor=c if filled else "white", edgecolor=c, lw=1.3, zorder=3, label=label)


def main(models):
    m = models[0]
    Q = pd.read_csv(TABLES / f"lit_debate_{m}_lit1200_requery.csv")
    U = Q[Q.unanimous.astype(bool)]
    hol = TABLES / f"lit_hollow_{m}_lit1200_auroc_defect.csv"
    ncol = 3 if hol.exists() else 2
    fig, axes = plt.subplots(1, ncol, figsize=(7.2 if ncol == 2 else 10.0, 3.2), constrained_layout=True)
    # a: deviation rates by regime: private vs public answer, after debate vs no communication
    ax = axes[0]
    groups = [("all", "All regimes")] + DOMS
    combos = [("debate", "private", style.TERRACOTTA, "o", True, "after debate, private"),
              ("debate", "public", style.TERRACOTTA, "s", False, "after debate, public"),
              ("reflect", "private", style.TEAL, "o", True, "no communication, private"),
              ("reflect", "public", style.TEAL, "s", False, "no communication, public")]
    for gi, (d, dl) in enumerate(groups):
        g0 = U if d == "all" else U[U.domain == d]
        for k, (c, q, col, mk, fill, lab) in enumerate(combos):
            g = g0[g0.cond == c]
            e, lo, hi = ci(g[f"dev_{q}"], g.cluster)
            hdot(ax, len(groups) - 1 - gi + 0.27 - 0.18 * k, e, lo, hi, col, mk, fill, label=lab if gi == 0 else None, s=24)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([l for _, l in groups][::-1])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Final answer differs from own last public answer (%)")
    ax.set_title(f"a  Asked privately vs publicly — {MNAME[m]}", loc="left")
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower left", ncols=4, fontsize=7.5)
    # b: debate - no communication difference, by regime, private vs public
    ax = axes[1]
    groups = [("all", "All regimes")] + DOMS
    for gi, (d, dl) in enumerate(groups):
        g = U if d == "all" else U[U.domain == d]
        A = g.groupby(["item_id", "cluster", "cond"])[["dev_private", "dev_public"]].mean().unstack("cond").dropna()
        cl = A.index.get_level_values("cluster")
        for k, (q, col, mk) in enumerate((("private", style.PLUM, "o"), ("public", style.SLATE, "s"))):
            e, lo, hi = ci(A[(f"dev_{q}", "debate")] - A[(f"dev_{q}", "reflect")], cl)
            hdot(ax, len(groups) - 1 - gi + (0.13 if k == 0 else -0.13), e, lo, hi, col, mk, k == 0,
                 label=(f"{q} answer" if gi == 0 else None))
    ax.axvline(0, color=style.SLATE, lw=0.8)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([l for _, l in groups][::-1])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Debate − no communication (percentage points)")
    ax.set_title("b  Effect of having debated", loc="left")
    ax.legend(loc="upper left", fontsize=7.5)
    if ncol == 3:
        ax = axes[2]
        A1 = pd.read_csv(hol)
        A1 = A1[A1.scale == "family_adjusted"].set_index("predictor")
        pr = [(p, l) for p, l in PRED if p in A1.index]
        for k, (p, l) in enumerate(pr):
            r = A1.loc[p]
            col = style.PLUM if p.startswith("neg_margin") else style.SLATE
            hdot(ax, len(pr) - 1 - k, r.auroc, r.lower, r.upper, col, "o", p != "neg_margin_observer")
        ax.axvline(0.5, color=style.SLATE, lw=0.8, ls=":")
        ax.set_yticks(range(len(pr)))
        ax.set_yticklabels([l for _, l in pr][::-1], fontsize=8)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Predicts which agents deviate in private (AUROC, within task family)")
        ax.set_title("c  What predicts private deviation", loc="left")
    style.finish(fig, axes, ROOT / "outputs/figures" / f"H4_audience_{m}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["Q9"])
