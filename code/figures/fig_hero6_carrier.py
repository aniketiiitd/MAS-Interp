"""Hero figure 6: what a peer's message is made of, and whether the parts add up to the whole.

Visual form: paired bars. For every scope the upper bar is the whole message as the peer wrote it; the lower bar is the
same message taken apart, the bare answer and the reasoning stacked end to end. If the two parts are independent
channels the lower bar reaches the upper one, and the eye checks that directly rather than reading three numbers off a
table. The right panel is the test stated as one quantity: the residual, parts minus whole, per scope and per model,
with the interval that decides it.

Reads the FULL-benchmark additivity tables. It read the half-benchmark ones until 2026-09-19, so panel b showed
residuals from half the data while the claim quoted the pooled figures.
Usage: python figures/fig_hero6_carrier.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/figures"))
import style  # noqa: E402
from style import plt  # noqa: E402

TABLES = ROOT / "outputs/tables"
OUT = ROOT / "outputs/figures/paper"
ANSWER, REASON, WHOLE = style.TERRACOTTA, style.TEAL, style.SLATE
SCOPES = [("all", "every item"), ("knowledge_reasoning", "knowledge"), ("safety_refusal", "safety"),
          ("values_moral_judgment", "values")]
MNAME = {"Q9": "Qwen3.5-9B", "L31": "Llama-3.1-8B", "G4": "Gemma-4-E4B"}
# Gemma ran the benchmark in one pass, so it has no _full2400 sibling
ADD = {"Q9": "lit_carrier_additivity_Q9_full2400.csv", "L31": "lit_carrier_additivity_L31_full2400.csv",
       "G4": "lit_carrier_additivity_G4.csv"}


def load(m):
    return pd.read_csv(TABLES / ADD[m]).set_index("scope")


def main():
    style.apply_paper()
    OUT.mkdir(parents=True, exist_ok=True)
    Q, L, G = load("Q9"), load("L31"), load("G4")
    fig, ax = style.strip(2, 3.15, [1.22, 0.92])

    # -- a: the whole message against its two parts, stacked, on Qwen -------------------------------------------------
    A = ax[0]
    h, gap = 0.30, 0.10
    ymax = 0.0
    for i, (key, lab) in enumerate(SCOPES):
        if key not in Q.index:
            continue
        r = Q.loc[key]
        y = -i * 1.30
        A.add_patch(Rectangle((0, y + gap / 2), r["full"], h, facecolor=WHOLE, edgecolor="none", zorder=3))
        A.add_patch(Rectangle((0, y - gap / 2 - h), r.answer_only, h, facecolor=ANSWER, edgecolor="none", zorder=3))
        A.add_patch(Rectangle((r.answer_only, y - gap / 2 - h), r.reasoning_only, h, facecolor=REASON,
                              edgecolor="white", lw=0.8, zorder=3))
        A.text(r["full"] + 0.08, y + gap / 2 + h / 2, f"{r['full']:.2f}", ha="left", va="center", fontsize=6.4,
               color=WHOLE, fontweight="bold")
        # A short segment cannot hold its own number: below about 0.45 logits the label is set outside it instead.
        if r.answer_only >= 0.45:
            A.text(r.answer_only / 2, y - gap / 2 - h / 2, f"{r.answer_only:.2f}", ha="center", va="center",
                   fontsize=6.0, color="white", fontweight="bold", zorder=5)
        else:
            A.text(-0.06, y - gap / 2 - h / 2, f"{r.answer_only:.2f}", ha="right", va="center", fontsize=6.0,
                   color=ANSWER, fontweight="bold", zorder=5)
        A.text(r.answer_only + r.reasoning_only / 2, y - gap / 2 - h / 2, f"{r.reasoning_only:.2f}", ha="center",
               va="center", fontsize=6.0, color="white", fontweight="bold", zorder=5)
        A.text(r.answer_only + r.reasoning_only + 0.08, y - gap / 2 - h / 2,
               f"{100 * r.reasoning_share:.0f}% reasoning", ha="left", va="center", fontsize=6.0, color=REASON)
        A.text(-0.12, y, lab, ha="right", va="center", fontsize=6.8, color=style.INK)
        ymax = min(ymax, y - gap / 2 - h)
    A.set_xlim(0, 4.55)
    A.set_ylim(ymax - 0.30, 0.90)
    A.set_yticks([])
    A.grid(axis="y", visible=False)
    A.spines["left"].set_visible(False)
    A.set_xlabel("how far the reader moves toward the peer (logits)")
    A.set_title("a  Qwen3.5-9B: the whole message, and its two parts", loc="left")
    # One sentence in the headroom above the first pair, rather than marks among the bars: every earlier attempt at
    # labelling the segments in place collided with the numbers already sitting in them.
    A.text(0, 0.60, "each pair: the message as the peer wrote it, then the same message taken apart",
           ha="left", va="center", fontsize=6.0, color=style.INK)
    A.text(0, 0.60, "                                                                ", ha="left", va="center",
           fontsize=6.0, color=style.INK)

    # -- b: the test itself, parts minus whole ------------------------------------------------------------------------
    B = ax[1]
    rows = []
    MCOL = {"Q9": WHOLE, "L31": style.PLUM, "G4": style.TEAL}
    MMK = {"Q9": "o", "L31": "s", "G4": "D"}
    for m, T in (("Q9", Q), ("L31", L), ("G4", G)):
        for key, lab in SCOPES:
            if key in T.index:
                rows.append((m, lab, T.loc[key]))
    for j, (m, lab, r) in enumerate(rows):
        y = len(rows) - 1 - j
        c = MCOL[m]
        B.plot([r.res_lo, r.res_hi], [y, y], color=c, lw=1.3, solid_capstyle="round", zorder=3)
        B.plot([r.residual], [y], marker=MMK[m], ms=3.4, color=c, zorder=4,
               markeredgecolor="white", markeredgewidth=0.5)
    B.axvline(0, color=style.INK, lw=0.8, zorder=2)
    B.set_yticks(range(len(rows)))
    B.set_yticklabels([lab for _, lab, _ in rows][::-1], fontsize=6.4)
    B.grid(axis="y", visible=False)
    B.set_ylim(-0.7, len(rows) - 0.3)
    bl, bh2 = B.get_xlim()
    B.set_xlim(bl - 0.06 * (bh2 - bl), bh2)   # keep the widest interval clear of the spine
    B.set_xlabel("parts minus whole (logits)")
    B.set_title("b  Do the parts add up?", loc="left")
    # name each model once, beside its own block of rows
    n = len(SCOPES)
    for k in (1, 2):
        B.axhline(k * n - 0.5, color=style.GRID, lw=0.7, zorder=1)
    lo, hi = B.get_xlim()
    # Both names sit at the same left margin: one left and one right read as an accident rather than a pairing.
    for y, m, c in ((len(rows) - 0.75, "Q9", WHOLE), (2 * n - 0.75, "L31", style.PLUM),
                    (n - 0.75, "G4", style.TEAL)):
        B.text(lo + 0.03 * (hi - lo), y, MNAME[m], ha="left", va="center", fontsize=6.4, color=c, fontweight="bold")
    # In data coordinates these sat on the last row once a third model was added. Axes fractions put them under the
    # plot where they belong, and they are the key to reading the panel: which side of zero a model falls on.
    B.text(0.5, -0.115, "parts fall short", transform=B.transAxes, ha="right", va="top", fontsize=6.0,
           color=style.MUTED)
    B.text(0.5, -0.115, "   parts overshoot", transform=B.transAxes, ha="left", va="top", fontsize=6.0,
           color=style.MUTED)

    style.finish_paper(fig, ax, str(OUT / "fig_hero6_carrier"))
    print("wrote", OUT / "fig_hero6_carrier.png")


if __name__ == "__main__":
    main()
