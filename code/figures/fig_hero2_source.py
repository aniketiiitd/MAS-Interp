"""Hero figure 2: what looks like respect for authority is a guess about reliability.

Visual form: the experiment IS a 2 x 3 crossed design (status x track record), so panel a draws it as a 2 x 3 matrix.
Each cell holds the shift toward the message relative to the identical message with no source description, with its
95% CI, tinted on the paper's diverging scale (terracotta = trusted more, teal = trusted less). The two marginal
brackets carry the interchange estimates: swapping the record moves the decision 0.56 logits, swapping the status 0.08.
Panel b is a slopegraph of the magnitude check: the record contrast shrunk to the status contrast's size, and the
status contrast amplified to the record's size, for both models. Even at equal size the record wins.

Reads outputs/tables/lit_source_Q9_cells.csv and lit_source_{Q9,L31}_magnitude.csv.
Usage: python figures/fig_hero2_source.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/figures"))
import style  # noqa: E402
from style import plt  # noqa: E402

TABLES = ROOT / "outputs/tables"
OUT = ROOT / "outputs/figures/paper"
MNAME = {"Q9": "Qwen3.5-9B", "L31": "Llama-3.1-8B", "G4": "Gemma-4-E4B"}
# Gemma is the model that contradicted this claim's first form, so a figure without it shows the
# evidence for the claim and not the evidence against its predecessor.
MODELS = [("Q9", "-", "o"), ("L31", (0, (2.5, 1.5)), "s"), ("G4", (0, (1, 1.2)), "^")]
REC, STA = style.TEAL, style.PLUM
ROWS = [("prof", "tenured\nprofessor"), ("ord", "ordinary member\nof the public")]
COLS = [("good", "strong"), ("unst", "unknown"), ("bad", "poor")]


def fmt(v):
    return f"{v:+.2f}".replace("-", "−")


def matrix(ax, m="Q9"):
    C = pd.read_csv(TABLES / f"lit_source_{m}_cells.csv").set_index("source")
    MG = pd.read_csv(TABLES / f"lit_source_{m}_magnitude.csv").set_index("arm")
    vmax = 0.55
    for ri, (s, _) in enumerate(ROWS):
        for ci, (r, _) in enumerate(COLS):
            row = C.loc[f"{s}_{r}"]
            t = 0.5 + 0.5 * np.clip(row.est / vmax, -1, 1) * 0.72  # keep the tint light enough for ink text
            face = style.DIVERGING(t)
            y = 1 - ri
            ax.add_patch(FancyBboxPatch((ci - 0.46, y - 0.42), 0.92, 0.84, boxstyle="round,pad=0,rounding_size=0.05",
                                        facecolor=face, edgecolor="white", lw=1.2, zorder=2))
            ax.text(ci, y + 0.10, fmt(row.est), ha="center", va="center", fontsize=10.5, fontweight="bold",
                    color=style.INK, zorder=3)
            ax.text(ci, y - 0.19, f"[{fmt(row.lower)}, {fmt(row.upper)}]", ha="center", va="center", fontsize=5.6,
                    color=style.SLATE, zorder=3)
    # the cell that carries the argument
    pb = C.loc["prof_bad"]
    y_prof = 1 - [r for r, _ in ROWS].index("prof")   # the row the annotation actually describes
    ax.add_patch(FancyBboxPatch((2 - 0.46, y_prof - 0.42), 0.92, 0.84, boxstyle="round,pad=0,rounding_size=0.05",
                                facecolor="none", edgecolor=style.INK, lw=1.3, zorder=4))
    ax.annotate(f"a professor with a poor record is trusted\n{abs(pb.est):.2f} logits LESS than no source at all",
                xy=(2.42, y_prof), xytext=(3.55, -0.55), ha="center", va="top", fontsize=6.0, color=style.INK,
                arrowprops=dict(arrowstyle="-", color=style.INK, lw=0.6, shrinkA=1, shrinkB=1), zorder=5)
    # marginal brackets: the interchange estimates from the magnitude table
    rec = MG.loc["record swap, as written"]
    sta = MG.loc["status swap, as written"]
    # all three models on the brackets: the matrix is Qwen's, but the interchange estimates are the numbers the
    # claim rests on and they differ by a factor of seventy across models
    allm = {mm: pd.read_csv(TABLES / f"lit_source_{mm}_magnitude.csv").set_index("arm") for mm, _, _ in MODELS}
    rec3 = " / ".join(f"{allm[mm].loc['record swap, as written'].est:.2f}" for mm, _, _ in MODELS)
    sta3 = " / ".join(f"{allm[mm].loc['status swap, as written'].est:.2f}" for mm, _, _ in MODELS)
    yb = 1.88   # clear of the column headers, which sit at 1.47 and stand about 0.15 tall
    ax.annotate("", xy=(2.3, yb), xytext=(-0.3, yb),
                arrowprops=dict(arrowstyle="<->", color=REC, lw=0.9, shrinkA=0, shrinkB=0, mutation_scale=7))
    ax.text(1.0, yb + 0.05, f"swap the record: {rec3} logits", ha="center", va="bottom", fontsize=6.6,
            color=REC, fontweight="bold")
    xb = 2.66
    ax.annotate("", xy=(xb, 0.92), xytext=(xb, 0.08),
                arrowprops=dict(arrowstyle="<->", color=STA, lw=0.9, shrinkA=0, shrinkB=0, mutation_scale=7))
    ax.text(xb + 0.08, 0.5, f"swap the status:\n{sta3}", ha="left", va="center", fontsize=6.6, color=STA,
            fontweight="bold")
    for ci, (_, lab) in enumerate(COLS):
        ax.text(ci, 1.47, lab, ha="center", va="bottom", fontsize=7.0, color=style.INK)
    ax.text(-0.55, 1.47, "record:", ha="right", va="bottom", fontsize=7.0, color=style.SLATE, style="italic")
    for ri, (_, lab) in enumerate(ROWS):
        ax.text(-0.55, 1 - ri, lab, ha="right", va="center", fontsize=7.0, color=style.INK, linespacing=1.1)
    ax.set_xlim(-1.55, 3.65)
    ax.set_ylim(-1.15, 2.55)
    ax.set_aspect("equal", adjustable="datalim", anchor="C")
    ax.axis("off")
    ax.set_title("a  Qwen3.5-9B: shift vs. no source described (logits)", loc="left")


def slope(ax):
    """Each model normalised to its own record effect, so three models fit one axis.

    Raw logits cannot carry all three: Gemma's record swap is 1.91 against Qwen's 0.56 and Llama's 0.88, so
    plotting them together would flatten the two smaller models into the baseline. Dividing by each model's own
    record effect asks the question the claim actually asks -- at equal signal magnitude, how much of the record's
    pull does the status reach -- and is the same quantity the claim quotes as a ratio.
    """
    arms = [("record swap, as written", "record, shrunk to status magnitude", REC, "record"),
            ("status swap, as written", "status, amplified to record magnitude", STA, "status")]
    xs = [0, 1]
    labels = []
    for mm, ls, mk in MODELS:
        MG = pd.read_csv(TABLES / f"lit_source_{mm}_magnitude.csv").set_index("arm")
        base = float(MG.loc["record swap, as written"].est)
        for a0, a1, c, name in arms:
            r0, r1 = MG.loc[a0], MG.loc[a1]
            ys = [float(r0.est) / base, float(r1.est) / base]
            ax.plot(xs, ys, color=c, ls=ls, lw=1.2, zorder=3)
            for x, r in zip(xs, (r0, r1)):
                ax.plot([x, x], [float(r.lower) / base, float(r.upper) / base], color=c, lw=0.7, zorder=3)
            ax.plot(xs, ys, marker=mk, ms=3.6, color=c, ls="", zorder=4,
                    markerfacecolor=c if mm == "Q9" else "white", markeredgewidth=0.9)
            if name == "status":
                labels.append((1.08, ys[1], f"{ys[1]:.2f}  {MNAME[mm].split('-')[0].split('3.5')[0].split('3.1')[0]}",
                               c, "left"))
    # only the status end is labelled: the record end is 1.00 by construction in every model
    L = sorted(labels, key=lambda l: l[1])
    ys = [l[1] for l in L]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + 0.085)
    for (x, _, t, c, ha), y in zip(L, ys):
        ax.text(x, y, t, ha=ha, va="center", fontsize=6.2, color=c)
    ax.axhline(1.0, color=style.MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.text(-0.30, 1.09, "the record, as written", ha="left", va="bottom", fontsize=5.6, color=style.MUTED)
    ax.set_xlim(-0.35, 2.25)
    ax.set_ylim(-0.03, 1.22)
    ax.set_xticks(xs)
    ax.set_xticklabels(["as written", "at equal\nmagnitude"], fontsize=6.6)
    ax.set_ylabel("share of the record's own effect")
    ax.grid(axis="x", visible=False)
    ax.spines["left"].set_visible(True)
    ax.set_title("b  Not a matter of signal size", loc="left")
    ax.text(0.0, -0.20, "solid Qwen, dashed Llama, dotted Gemma", transform=ax.transAxes,
            fontsize=5.4, color=style.SLATE, ha="left", va="top")


def main():
    style.apply_paper()
    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(style.TEXTWIDTH, 2.05), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0])
    ax = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]
    matrix(ax[0])
    slope(ax[1])
    style.finish_paper(fig, ax, str(OUT / "fig_hero2_source"))
    print("wrote", OUT / "fig_hero2_source.png")


if __name__ == "__main__":
    main()
