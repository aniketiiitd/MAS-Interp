"""Emotion and moral concept levels across twelve turns of debate, against the matched no-communication run.

Every value is debate minus no-communication at the same turn, read at the point just before an agent writes, so
anything that comes merely from the context getting longer cancels. The grey band is what the same directions refit
with the concept labels shuffled inside each topic produce on the same states: a concept counts as moving only when it
leaves that band. Concepts that never leave it are drawn faint, because a figure that only shows the movers would be
choosing its own evidence.

Usage: python figures/fig_turn_concepts.py [MODEL] [DATASET]
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/figures"))
import style  # noqa: E402
from style import plt  # noqa: E402

TABLES = ROOT / "outputs/tables"
OUT = ROOT / "outputs/figures/paper"
TAGS = [("emotion_stories", "emotion"), ("moral_stories", "moral")]
HOT, COOL = style.TERRACOTTA, style.TEAL


def panel(ax, T, title, n_label=6):
    """One tag: every concept as a faint line, the ones that leave the shuffled-label band named."""
    # A missing table arrives as an empty frame with no columns at all, so the column check comes before the filter.
    A = T[T.scope == "all"] if len(T) and "scope" in T.columns else pd.DataFrame()
    if not len(A):
        ax.text(0.5, 0.5, "not yet available", transform=ax.transAxes, ha="center", va="center",
                fontsize=7, color=style.MUTED)
        ax.set_title(title, loc="left")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        return 0
    L = int(A.groupby("layer").moves.mean().idxmax())
    A = A[A.layer == L]
    band = A.groupby("round").null_band95.first()
    ax.fill_between(band.index, -band.values, band.values, color=style.GRID, alpha=0.55, lw=0,
                    label="shuffled-label band", zorder=1)
    ax.axhline(0, color=style.INK, lw=0.6, zorder=2)

    mover = A.groupby("concept").moves.any()
    movers = list(mover[mover].index)
    final = A[A["round"] == A["round"].max()].set_index("concept").debate_minus_solo
    named = sorted(movers, key=lambda c: -abs(final.get(c, 0)))[:n_label]
    ends = []
    for c, g in A.groupby("concept"):
        g = g.sort_values("round")
        if c in named:
            col = HOT if final.get(c, 0) > 0 else COOL
            ax.plot(g["round"], g.debate_minus_solo, color=col, lw=1.3, zorder=4)
            ends.append((float(g.debate_minus_solo.iloc[-1]), c, col, float(g["round"].iloc[-1])))
        else:
            ax.plot(g["round"], g.debate_minus_solo, color=style.MUTED, lw=0.4, alpha=0.32, zorder=3)
    # push the end labels apart on a minimum pitch, keeping their order, so several concepts ending close together
    # stay readable
    ends.sort(key=lambda e: -e[0])
    span = float(A.debate_minus_solo.max() - A.debate_minus_solo.min()) or 1.0
    pitch = span / 13.0
    ys = [e[0] for e in ends]
    for i in range(1, len(ys)):
        ys[i] = min(ys[i], ys[i - 1] - pitch)
    for (y0, c, col, x0), y in zip(ends, ys):
        ax.plot([x0 + 0.06, x0 + 0.20], [y0, y], color=col, lw=0.5, alpha=0.7, zorder=4)
        ax.text(x0 + 0.24, y, c, fontsize=5.8, color=col, va="center", ha="left")
    ax.set_xlabel("turn")
    ax.set_xlim(A["round"].min() - 0.2, A["round"].max() + max(1.6, 0.30 * A["round"].max()))
    ax.set_title(f"{title}  (layer {L}; {len(movers)} of {A.concept.nunique()} leave the band)", loc="left")
    return len(movers)


def main(m="Q9", ds="release_600_t12"):
    style.apply_paper()
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = style.strip(2, 2.25)
    total = 0
    for i, (tag, lab) in enumerate(TAGS):
        p = TABLES / f"lit_turn_concepts_{tag}_{m}_{ds}.csv"
        T = pd.read_csv(p) if p.exists() else pd.DataFrame()
        total += panel(ax[i], T, lab)
    ax[0].set_ylabel("debate minus no communication")
    ax[0].legend(loc="upper left", fontsize=6.0, frameon=False)
    style.finish_paper(fig, ax, str(OUT / f"fig_turn_concepts_{m}"))
    print(f"wrote {OUT}/fig_turn_concepts_{m}.png  ({total} concepts leave the band in total)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Q9",
         sys.argv[2] if len(sys.argv) > 2 else "release_600_t12")
