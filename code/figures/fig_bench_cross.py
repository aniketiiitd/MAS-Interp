"""MAD Bench: debate against a matched non-communicating pool, resolved by answer regime, across models.

Every quantity is item-level over the 761 items with an evaluation target, computed by analysis/lit_debate.py for each
model separately; nothing is re-derived here. Models that have no vote table yet are skipped, so this renders at any
point during the replication chains.

  a  accuracy, debate majority minus majority vote of the initial answers
  b  accuracy, debate minus the matched no-communication pool
  c  ends unanimous on a wrong answer, debate minus no communication

Usage: python figures/fig_bench_cross.py [MODEL ...]
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "figures"))
import style  # noqa: E402
from common import TABLES  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "outputs/figures/paper"
S_, M_ = style.SLATE, style.MUTED
MNAME = {"Q9": "Qwen3.5-9B", "L31": "Llama-3.1-8B", "G4": "Gemma-4-E4B"}
MMARK = {"Q9": ("o", style.TERRACOTTA), "L31": ("s", style.TEAL), "G4": ("D", style.PLUM)}
ROWS = [("all", "all regimes"), ("knowledge_reasoning", "Knowledge"), ("safety_refusal", "Safety"),
        ("values_moral_judgment", "Values")]
PANELS = [("debate - vote", "a  Debate − vote"), ("debate - reflect", "b  Debate − no communication"),
          ("wrong_unan debate - reflect", "c  Wrong unanimity, debate − no comm.")]


def main(models):
    style.apply_paper()
    T = {}
    for m in models:
        try:
            T[m] = pd.read_csv(TABLES / f"lit_debate_{m}_lit1200_vote.csv")
        except Exception:
            pass
    if not T:
        print("no vote tables yet")
        return
    have = [m for m in models if m in T]
    fig, ax = style.strip(3, 1.75, [1.12, 0.94, 0.94])
    for pi, (metric, title) in enumerate(PANELS):
        for ri, (dom, _) in enumerate(ROWS):
            for k, m in enumerate(have):
                r = T[m][(T[m].domain == dom) & (T[m].metric == metric)]
                if not len(r):
                    continue
                r = r.iloc[0]
                mk, c = MMARK.get(m, ("o", S_))
                off = 0.22 - 0.22 * k if len(have) > 1 else 0.0
                y = len(ROWS) - 1 - ri + off
                ax[pi].plot([100 * r.lower, 100 * r.upper], [y, y], color=c, lw=1.0, solid_capstyle="round", zorder=2)
                ax[pi].scatter([100 * r.est], [y], s=15, marker=mk, facecolor=c, edgecolor=c, lw=0.9, zorder=3,
                               label=MNAME.get(m, m) if (pi == 0 and ri == 0) else None)
        ax[pi].axvline(0, color=S_, lw=0.6)
        ax[pi].set_yticks(range(len(ROWS)))
        ax[pi].set_yticklabels([l for _, l in ROWS][::-1] if pi == 0 else [])
        ax[pi].grid(axis="y", visible=False)
        ax[pi].set_ylim(-0.75, len(ROWS) - 0.4)
        ax[pi].set_xlabel("percentage points")
        ax[pi].set_title(title, loc="left")
    ax[0].legend(loc="lower left", fontsize=5.8, frameon=False, labelspacing=0.22, borderpad=0.1, handletextpad=0.35)
    style.finish_paper(fig, ax, f"{OUT}/fig_bench_cross")
    print(f"wrote {OUT}/fig_bench_cross.png for {have}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["Q9", "L31", "G4"])
