"""Hero figure 1: the answer is settled before the argument is written.

Visual form: two "writing timelines" (one per model). The x axis is how much of its own reasoning the agent has
written when it is stopped and forced to answer; the y axis is how often that forced answer already equals the answer
it will eventually give. Only agents that CHANGED their answer are plotted, in debate (has read its peers) and without
communication. The shaded wedge between the two curves is what having read the peers contributes, and the whole story
is its height at x = 0. Panel c repeats the x = 0 measurement with three different texts in front of the forced answer:
nothing, neutral filler of the same length, and another item's reasoning. The wedge survives filler and collapses under
the wrong reasoning.

Reads the FULL-benchmark commit curves for each model and lit_filler_Q9_full2400_summary.csv. It read the
half-benchmark tables until 2026-09-19, which meant the figure plotted 66.7%/20.3% for Qwen while the claim text
beside it quoted the pooled 65.4%/22.3%: two numbers for one quantity in one document.
Usage: python figures/fig_hero1_answer.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/figures"))
import style  # noqa: E402
from style import plt  # noqa: E402

TABLES = ROOT / "outputs/tables"
OUT = ROOT / "outputs/figures/paper"
MNAME = {"Q9": "Qwen3.5-9B", "L31": "Llama-3.1-8B", "G4": "Gemma-4-E4B"}
# Gemma ran the whole benchmark in one pass, so it has no "_full2400" suffix; the other two were pooled from halves
CURVES = {"Q9": "lit_commit_Q9_full2400_curves.csv", "L31": "lit_commit_L31_full2400_curves.csv",
          "G4": "lit_commit_G4_curves.csv"}
DEB, REF = style.TERRACOTTA, style.TEAL
WEDGE = "#D9D9D9"  # neutral: the wedge marks the gap, and a third hue only competed with the two curves


def curves(m):
    T = pd.read_csv(TABLES / CURVES[m])
    T = T[(T.domain == "all") & (T.metric == "argmax_is_final")]
    out = {}
    for g in ("changed answer in debate", "changed answer without communication", "kept answer in debate",
              "kept answer without communication"):
        d = T[T.group == g].sort_values("fraction")
        out[g] = (100 * d.fraction.to_numpy(), 100 * d.est.to_numpy(), 100 * d.lower.to_numpy(), 100 * d.upper.to_numpy())
    return out


def series(ax, x, y, lo, hi, c, mk, ls, label, z=4):
    ax.plot(x, y, color=c, ls=ls, marker=mk, ms=3.4, lw=1.3, label=label, zorder=z, clip_on=False,
            markeredgecolor="white", markeredgewidth=0.5)
    ax.vlines(x, lo, hi, color=c, lw=0.7, zorder=z - 1, alpha=0.9)


def timeline(ax, m, first, letter):
    C = curves(m)
    xd, yd, lod, hid = C["changed answer in debate"]
    xr, yr, lor, hir = C["changed answer without communication"]
    # the wedge: what having read the peers contributes to the answer before any own reasoning is on the page
    ax.fill_between(xd, yr, yd, color=WEDGE, alpha=0.75, lw=0, zorder=2)
    # agents that kept their answer are ~99% from the first token in both conditions: one reference line, not two curves
    yk = np.mean([C["kept answer in debate"][1], C["kept answer without communication"][1]])
    ax.axhline(yk, color=style.MUTED, lw=0.7, ls=(0, (1.5, 1.5)), zorder=1)
    ax.text(2, yk - 2.5, f"kept its answer: {yk:.0f}%", ha="left", va="top", fontsize=5.8, color=style.SLATE)
    series(ax, xr, yr, lor, hir, REF, "s", (0, (3.5, 1.5)), "changed, no peers seen")
    series(ax, xd, yd, lod, hid, DEB, "o", "-", "changed, read its peers")
    # the gap at zero, stated on the plot
    gap = yd[0] - yr[0]
    ax.annotate("", xy=(0, yd[0] - 3.0), xytext=(0, yr[0] + 3.0),
                arrowprops=dict(arrowstyle="<->", color=style.INK, lw=0.8, shrinkA=0, shrinkB=0, mutation_scale=7), zorder=6)
    # a blank lane left of x = 0 holds the three numbers that are the point of the figure
    ax.text(-3, (yd[0] + yr[0]) / 2, f"{gap:.0f} pts", ha="right", va="center", fontsize=6.6, fontweight="bold",
            color=style.INK, zorder=7)
    ax.text(-3, yd[0] + 0.5, f"{yd[0]:.1f}%", ha="right", va="bottom", fontsize=6.6, color=DEB, fontweight="bold")
    ax.text(-3, yr[0] - 0.5, f"{yr[0]:.1f}%", ha="right", va="top", fontsize=6.6, color=REF, fontweight="bold")
    ax.set_xlim(-27, 102)
    ax.set_ylim(0, 104)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([0, 50, 100])
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("own reasoning written (%)")
    if first:
        ax.set_ylabel("forced answer = eventual answer (%)")
    else:
        ax.set_yticklabels([])
    ax.set_title(f"{letter}  {MNAME[m]}", loc="left")


def prefixes(ax):
    S = pd.read_csv(TABLES / "lit_filler_Q9_full2400_summary.csv")
    S = S[S.group == "changed"]
    # short labels: at four panels the third one ran into the second at any readable size
    arms = [("frac0", "nothing"), ("filler", "neutral\nfiller"), ("cross_item", "another\nitem's\nreasoning")]
    for i, (arm, lab) in enumerate(arms):
        d = S[(S.cond == "debate") & (S.arm == arm)].iloc[0]
        r = S[(S.cond == "reflect") & (S.arm == arm)].iloc[0]
        gap = 100 * (d.est - r.est)
        ax.fill_between([i - 0.22, i + 0.22], 100 * r.est, 100 * d.est, color=WEDGE, alpha=0.75, lw=0, zorder=2)
        for row, c, mk in ((r, REF, "s"), (d, DEB, "o")):
            ax.plot([i, i], [100 * row.lower, 100 * row.upper], color=c, lw=0.7, zorder=3)
            ax.plot([i], [100 * row.est], marker=mk, ms=3.6, color=c, zorder=4, ls="", markeredgecolor="white",
                    markeredgewidth=0.5)
        ymid = 50 * (d.est + r.est)
        if gap > 8:
            ax.text(i, ymid, f"{gap:.0f} pts", ha="center", va="center", fontsize=6.4, fontweight="bold", color=style.INK,
                    zorder=5)
        else:
            ax.text(i + 0.26, ymid, f"{gap:.1f} pts", ha="left", va="center", fontsize=6.4, fontweight="bold", color=style.INK)
        ax.text(i, 100 * d.upper + 2.5, f"{100 * d.est:.1f}", ha="center", va="bottom", fontsize=5.8, color=DEB)
        ax.text(i, 100 * r.lower - 2.5, f"{100 * r.est:.1f}", ha="center", va="top", fontsize=5.8, color=REF)
    ax.set_xlim(-0.55, 2.95)
    ax.set_ylim(0, 104)
    ax.set_xticks(range(3))
    ax.set_xticklabels([l for _, l in arms], fontsize=5.4, linespacing=0.95)
    ax.set_yticks([0, 50, 100])
    ax.set_yticklabels([])
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("text before the answer")
    ax.set_title("d  Qwen3.5-9B, at 0%", loc="left")


def main():
    style.apply_paper()
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = style.strip(4, height=2.15, width_ratios=[1.0, 1.0, 1.0, 0.82])
    for i, (m, letter) in enumerate((("Q9", "a"), ("L31", "b"), ("G4", "c"))):
        timeline(ax[i], m, i == 0, letter)
    prefixes(ax[3])
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h[::-1], l[::-1], loc="outside lower center", ncols=2, frameon=False, handlelength=2.2, columnspacing=1.6,
               fontsize=6.6)
    style.finish_paper(fig, ax, str(OUT / "fig_hero1_answer"))
    print("wrote", OUT / "fig_hero1_answer.png")


if __name__ == "__main__":
    main()
