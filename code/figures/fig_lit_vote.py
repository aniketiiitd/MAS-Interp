"""Hero 5 figure: debate vs voting on MAD Bench's three answer regimes (analysis/lit_debate.py items table).
Usage: python figures/fig_lit_vote.py Q9 [L31 G4]
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
COND = [("vote", "majority vote of initial answers", style.SLATE, "o", False),
        ("reflect_maj", "3 rounds, no communication", style.TEAL, "s", True),
        ("debate_maj", "3 rounds of debate", style.TERRACOTTA, "o", True)]


def ci(x, c):
    e, lo, hi, *_ = fast_cluster_mean_ci(np.asarray(x, float), np.asarray(c))
    return 100 * e, 100 * lo, 100 * hi


def hdot(ax, y, e, lo, hi, c, m, filled, label=None, s=34):
    ax.plot([lo, hi], [y, y], color=c, lw=1.3, solid_capstyle="round", zorder=2)
    ax.scatter([e], [y], s=s, marker=m, facecolor=c if filled else "white", edgecolor=c, lw=1.3, zorder=3, label=label)


def main(models):
    n = len(models)
    fig, axes = plt.subplots(n, 2, figsize=(7.2, 2.7 * n + 0.5), constrained_layout=True, squeeze=False,
                             gridspec_kw=dict(width_ratios=[1.15, 1.0]))
    for mi, m in enumerate(models):
        I = pd.read_csv(TABLES / f"lit_debate_{m}_lit1200_items.csv")
        T = I[I.target.notna() & (I.kind != "numeric")].copy()
        for c in ("vote", "debate_maj", "reflect_maj"):
            T[f"acc_{c}"] = (T[c] == T.target).astype(float)
        # a: accuracy by regime
        ax = axes[mi, 0]
        for di, (dom, lab) in enumerate(DOMS):
            g = T[T.domain == dom]
            for ci_, (c, clab, col, mk, fill) in enumerate(COND):
                e, lo, hi = ci(g[f"acc_{c}"], g.cluster)
                hdot(ax, 2 - di + 0.22 - 0.22 * ci_, e, lo, hi, col, mk, fill, label=clab if (di == 0 and mi == 0) else None)
            ax.text(1.0, 2 - di, f" n={len(g)}", transform=ax.get_yaxis_transform(), va="center", fontsize=7, color=style.SLATE)
        ax.set_yticks(range(3))
        ax.set_yticklabels([l for _, l in DOMS][::-1])
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Final majority answer correct (%)")
        ax.set_title(("a  " if mi == 0 else "") + f"Accuracy by answer regime — {MNAME[m]}", loc="left")
        # b: entrenchment of a wrong initial majority
        ax = axes[mi, 1]
        W = T[T.vote.notna() & (T.vote != T.target)].copy()
        for c in ("debate", "reflect"):
            W[f"wu_{c}"] = (W[f"{c}_unan"].astype(bool) & (W[f"{c}_maj"] != W.target)).astype(float)
        groups = [("all", "All regimes", W)] + [(d, l, W[W.domain == d]) for d, l in DOMS]
        for gi, (_, lab, g) in enumerate(groups):
            if len(g) < 5:
                continue
            for k, (c, col, mk) in enumerate((("reflect", style.TEAL, "s"), ("debate", style.TERRACOTTA, "o"))):
                e, lo, hi = ci(g[f"wu_{c}"], g.cluster)
                hdot(ax, len(groups) - 1 - gi + (0.13 if k == 1 else -0.13), e, lo, hi, col, mk, True)
            ax.text(1.0, len(groups) - 1 - gi, f" n={len(g)}", transform=ax.get_yaxis_transform(), va="center", fontsize=7, color=style.SLATE)
        ax.set_yticks(range(len(groups)))
        ax.set_yticklabels([l for _, l, _ in groups][::-1])
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Ends unanimous on the wrong answer (%)")
        ax.set_title(("b  " if mi == 0 else "") + "When the initial majority was wrong", loc="left")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncols=3, fontsize=8)
    style.finish(fig, axes, ROOT / "outputs/figures" / f"H5_debate_vs_vote_{'_'.join(models)}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["Q9"])
