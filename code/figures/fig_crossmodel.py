"""The four live results side by side across models, as each replication lands.

Reads whatever per-model tables exist and silently skips the rest, so it can be run at any point during the L31 and
Gemma chains. Every panel is the same quantity computed by the same script for each model; nothing is re-derived here.

Panels
  a  when the answer is chosen   fraction-0 read-out for answer changes, debate vs no communication
  b  status vs track record      the record:status advantage at matched residual magnitude (log scale at 1 = no advantage)
  c  read-out site               AUROC for private disagreement from the written answer token vs a fresh re-read
  d  ownership of the words      extra pull when identical text is labelled the agent's own, with the position control

Usage: python figures/fig_crossmodel.py [MODEL ...]      (default: Q9 L31 G4)
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


def read(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def hdot(ax, y, e, lo, hi, c, m, filled=True, s=15):
    if not np.isfinite(e):
        return
    ax.plot([lo, hi], [y, y], color=c, lw=1.0, solid_capstyle="round", zorder=2)
    ax.scatter([e], [y], s=s, marker=m, facecolor=c if filled else "white", edgecolor=c, lw=0.9, zorder=3)


def main(models):
    style.apply_paper()
    have = [m for m in models if read(TABLES / f"lit_commit_{m}_curves.csv") is not None]
    if not have:
        print("no model tables yet")
        return
    fig, ax = style.strip(4, 1.9, [1.02, 0.78, 0.92, 0.84])
    yy = {m: len(have) - 1 - k for k, m in enumerate(have)}

    for m in have:
        T = read(TABLES / f"lit_commit_{m}_curves.csv")
        for grp, c, mk, off in (("changed answer in debate", style.TERRACOTTA, "o", 0.16),
                                ("changed answer without communication", style.TEAL, "s", -0.16)):
            r = T[(T.group == grp) & (T.domain == "all") & (T.metric == "argmax_is_final") & (T.fraction == 0.0)]
            if len(r):
                r = r.iloc[0]
                hdot(ax[0], yy[m] + off, 100 * r.est, 100 * r.lower, 100 * r.upper, c, mk)
    ax[0].set_xlim(0, 100)
    ax[0].set_xlabel("decided before writing (%)")
    ax[0].set_title("a  When the answer is chosen", loc="left")
    ax[0].text(0.97, 0.03, "debate / no comm.", transform=ax[0].transAxes, ha="right", va="bottom", fontsize=5.4, color=M_)

    for m in have:
        M = read(TABLES / f"lit_source_{m}_magnitude.csv")
        if M is None:
            continue
        M = M.set_index("arm")
        for key, c, mk, off in (("record:status advantage at the record magnitude", style.TERRACOTTA, "o", 0.16),
                                ("record:status advantage at the status magnitude", style.PLUM, "D", -0.16)):
            if key in M.index:
                r = M.loc[key]
                hdot(ax[1], yy[m] + off, r.est, r.lower, r.upper, c, mk)
    ax[1].axvline(1.0, color=S_, lw=0.6, ls=(0, (3, 2)))
    ax[1].set_xlabel("record : status (\u00d7)")
    ax[1].set_title("b  Status vs record", loc="left")
    ax[1].text(0.97, 0.03, "at record / status size", transform=ax[1].transAxes, ha="right", va="bottom", fontsize=5.4, color=M_)

    for m in have:
        A = read(TABLES / f"lit_hollow_{m}_lit1200_auroc_defect.csv")
        if A is None:
            continue
        A = A[A.scale == "family_adjusted"].set_index("predictor")
        for key, c, mk, off in (("neg_margin", style.TERRACOTTA, "o", 0.16), ("neg_margin_part", style.TEAL, "s", -0.16)):
            if key in A.index:
                r = A.loc[key]
                hdot(ax[2], yy[m] + off, r.auroc, r.lower, r.upper, c, mk)
    ax[2].axvline(0.5, color=S_, lw=0.6, ls=(0, (3, 2)))
    ax[2].set_xlabel("AUROC (disagreement)")
    ax[2].set_title("c  Where you read it", loc="left")
    ax[2].text(0.97, 0.03, "written token / fresh read", transform=ax[2].transAxes, ha="right", va="bottom", fontsize=5.4, color=M_)

    for m in have:
        O = read(TABLES / f"lit_frames_{m}_lit1200_own.csv")
        if O is None:
            continue
        O = O[O.domain == "all"].set_index("metric")
        for key, c, mk, off in (("delta_own", style.TERRACOTTA, "o", 0.16),
                                ("position_last_minus_first", M_, "s", -0.16)):
            if key in O.index:
                r = O.loc[key]
                hdot(ax[3], yy[m] + off, r.est, r.lower, r.upper, c, mk)
    ax[3].axvline(0, color=S_, lw=0.6)
    ax[3].set_xlabel("pull when labelled own")
    ax[3].set_title("d  Whose words", loc="left")
    ax[3].text(0.97, 0.03, "label swap / position", transform=ax[3].transAxes, ha="right", va="bottom", fontsize=5.4, color=M_)

    for k, a in enumerate(ax):
        a.set_yticks(range(len(have)))
        a.set_yticklabels([MNAME.get(m, m) for m in have][::-1] if k == 0 else [])
        a.grid(axis="y", visible=False)
        a.set_ylim(-0.6, len(have) - 0.4)
    style.finish_paper(fig, ax, f"{OUT}/fig_crossmodel")
    print(f"wrote {OUT}/fig_crossmodel.png for {have}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["Q9", "L31", "G4"])
