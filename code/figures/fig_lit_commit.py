"""Figure: debate-induced answer changes are decided before the agent writes its reasoning (analysis/lit_commit.py tables).
Usage: python figures/fig_lit_commit.py Q9
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "figures"))
from common import *  # noqa
import style
from style import plt

style.apply()
MNAME = {"Q9": "Qwen3.5-9B", "L31": "Llama-3.1-8B", "G4": "Gemma-4-E4B"}
GROUPS = [("changed answer in debate", style.TERRACOTTA, "o", "-"), ("changed answer without communication", style.TEAL, "s", "-"),
          ("kept answer in debate", style.TERRACOTTA, "o", ":"), ("kept answer without communication", style.TEAL, "s", ":")]
DOMS = [("knowledge_reasoning", "Knowledge"), ("safety_refusal", "Safety"), ("values_moral_judgment", "Values")]
FLAB = {"calibrated_abstention": "calibrated abstention", "ethical_framework": "ethical frameworks", "moral_tradeoffs": "moral trade-offs",
        "public_policy": "public-policy debate", "social_influence": "persuasion", "strategic_deception": "strategic deception",
        "over_refusal": "over-refusal controls", "misconception_resistance": "misconception resistance"}


def main(m):
    T = pd.read_csv(TABLES / f"lit_commit_{m}_curves.csv")
    F = pd.read_csv(TABLES / f"lit_commit_{m}_family_fraction0.csv")
    fig = plt.figure(figsize=(7.2, 5.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])
    ax = fig.add_subplot(gs[0, 0])
    for g, c, mk, ls in GROUPS:
        d = T[(T.group == g) & (T.domain == "all") & (T.metric == "argmax_is_final")].sort_values("fraction")
        ax.plot(100 * d.fraction, 100 * d.est, color=c, ls=ls, marker=mk, ms=4, lw=1.5, label=g)
        ax.fill_between(100 * d.fraction, 100 * d.lower, 100 * d.upper, color=c, alpha=0.15, lw=0)
    ax.set_xlabel("Share of its own written reasoning the agent has seen (%)")
    ax.set_ylabel("Forced answer = answer it finally wrote (%)")
    ax.set_ylim(0, 103)
    ax.set_title(f"a  When is the final answer chosen? ({MNAME[m]})", loc="left")
    ax.legend(loc="lower right", fontsize=7.2)
    ax = fig.add_subplot(gs[0, 1])
    rows = [("all", "All regimes")] + DOMS
    for ri, (d, lab) in enumerate(rows):
        for k, (g, c, mk) in enumerate((("changed answer in debate", style.TERRACOTTA, "o"), ("changed answer without communication", style.TEAL, "s"))):
            r = T[(T.group == g) & (T.domain == d) & (T.metric == "argmax_is_final") & (T.fraction == 0.0)].iloc[0]
            y = len(rows) - 1 - ri + (0.13 if k == 0 else -0.13)
            ax.plot([100 * r.lower, 100 * r.upper], [y, y], color=c, lw=1.3)
            ax.scatter([100 * r.est], [y], color=c, marker=mk, s=32, zorder=3)
            ax.text(101, y, f"n={int(r.n)}", fontsize=6.8, va="center", color=style.SLATE)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([l for _, l in rows][::-1])
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 112)
    ax.set_xlabel("New answer already chosen before writing (%)")
    ax.set_title("b  Answer changes decided on reading", loc="left")
    ax = fig.add_subplot(gs[1, :])
    fams = [f for f in FLAB if len(F[(F.family == f) & F.est.notna()]) == 2]
    fams = sorted(fams, key=lambda f: -F[(F.family == f) & (F.cond == "debate")].est.iloc[0])
    for fi, f in enumerate(fams):
        for k, (cond, c, mk) in enumerate((("debate", style.TERRACOTTA, "o"), ("reflect", style.TEAL, "s"))):
            r = F[(F.family == f) & (F.cond == cond)].iloc[0]
            x = fi + (-0.12 if k == 0 else 0.12)
            ax.plot([x, x], [100 * r.lower, 100 * r.upper], color=c, lw=1.3)
            ax.scatter([x], [100 * r.est], color=c, marker=mk, s=30, zorder=3,
                       label=("changed answer in debate" if cond == "debate" else "changed answer without communication") if fi == 0 else None)
    ax.set_xticks(range(len(fams)))
    ax.set_xticklabels([FLAB[f] for f in fams], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Decided before writing (%)")
    ax.set_ylim(-3, 103)
    ax.grid(axis="x", visible=False)
    ax.set_title("c  By MAD Bench task family (families with answer changes in both conditions)", loc="left")
    ax.legend(loc="upper right", fontsize=7.2, ncols=2)
    style.finish(fig, fig.axes, ROOT / "outputs/figures" / f"H6_decided_before_reasoning_{m}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Q9")
