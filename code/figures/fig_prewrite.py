"""The reading-state transplant, as a horizontal strip at the ICLR text width.
Usage: python figures/fig_prewrite.py Q9
Reads outputs/tables/lit_prewrite_<M>_{arms,layers}.csv (written by analysis/lit_prewrite.py); no recomputation.
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

style.apply_paper()
OUT = ROOT / "outputs/figures/paper"
OUT.mkdir(parents=True, exist_ok=True)
MNAME = {"Q9": "Qwen3.5-9B", "L31": "Llama-3.1-8B", "G4": "Gemma-4-E4B"}
NL = {"Q9": 32, "L31": 32, "G4": 42}
T_, R_, P_, S_, M_ = style.TERRACOTTA, style.TEAL, style.PLUM, style.SLATE, style.MUTED

LINES = [("debate", R_, "-", "o", "peers' state"), ("debate_x3", R_, "--", "s", "peers' state, 3x"),
         ("cross", P_, "-", "^", "another item's state"), ("randdiff", M_, ":", "v", "random, same size")]
ROWS = [("debate_ref", S_, "real debate context"), ("debate_x3", R_, "peers' state, 3x"), ("debate", R_, "peers' state"),
        ("cross", P_, "another item's state"), ("randdiff_x3", M_, "random, same size, 3x"), ("randdiff", M_, "random, same size")]


def get(T, kind, msr, col="est"):
    r = T[(T.kind == kind) & (T.measure == msr)]
    return float(r[col].iloc[0]) if len(r) else np.nan


def main(m):
    A = pd.read_csv(TABLES / f"lit_prewrite_{m}_arms.csv")
    B = pd.read_csv(TABLES / f"lit_prewrite_{m}_layers.csv")
    best = int(B.layer.iloc[0])
    fig, ax = style.strip(3, height=1.85, width_ratios=[1.18, 1.0, 0.82])

    # (a) effect on the peers' answer, by the layer the state is written into
    a = ax[0]
    D = A[A.measure == "d_p_target"]
    ref = D[D.kind == "debate_ref"]
    if len(ref):
        a.axhspan(float(ref.lower.iloc[0]), float(ref.upper.iloc[0]), color=S_, alpha=0.12, lw=0)
        a.axhline(float(ref.est.iloc[0]), color=S_, lw=0.9, ls=(0, (4, 2)), zorder=1)
        a.text(0.985, float(ref.est.iloc[0]), "real debate", color=S_, fontsize=5.8, ha="right", va="top",
               transform=a.get_yaxis_transform(), clip_on=False)
    a.axhline(0, color=style.INK, lw=0.6, zorder=1)
    for kind, c, ls, mk, lab in LINES:
        g = D[D.kind == kind].dropna(subset=["layer"]).sort_values("layer")
        if not len(g):
            continue
        a.fill_between(g.layer, g.lower, g.upper, color=c, alpha=0.13, lw=0)
        a.plot(g.layer, g.est, color=c, ls=ls, lw=1.0, marker=mk, ms=2.6, label=lab)
    a.set_xticks(sorted(D.dropna(subset=["layer"]).layer.unique()))
    a.set_xlabel("layer the state is written into" + (f" (of {NL[m]})" if m in NL else ""))
    a.set_ylabel("shift toward the peers' answer")
    a.legend(loc="upper left", fontsize=5.5, frameon=False, handlelength=1.2, labelspacing=0.25, borderaxespad=0.1,
             handletextpad=0.4)

    # (b) at the layer where the transplant does most, every arm side by side
    b = ax[1]
    b.axvline(0, color=style.INK, lw=0.6, zorder=1)
    labs = []
    for j, (kind, c, lab) in enumerate(ROWS):
        y = len(ROWS) - 1 - j
        e, lo, hi = get(B, kind, "d_p_target"), get(B, kind, "d_p_target", "lower"), get(B, kind, "d_p_target", "upper")
        if np.isfinite(e):
            b.plot([lo, hi], [y, y], color=c, lw=1.0, solid_capstyle="round", zorder=2)
            b.scatter([e], [y], s=15, marker="o", facecolor=c if "random" not in lab else "white", edgecolor=c, lw=0.9, zorder=3)
        labs.append(lab)
    b.set_yticks(range(len(labs)))
    b.set_yticklabels(labs[::-1], fontsize=6.0)
    b.grid(axis="y", visible=False)
    b.set_ylim(-0.6, len(labs) - 0.4)
    b.set_xlabel(f"shift toward the peers' answer (layer {best})")

    # (c) the same responses: does the reasoning refer to the other agents at all?
    c_ = ax[2]
    items = [("none", M_, "no-communication"), ("debate", R_, "transplant"), ("debate_ref", S_, "real debate")]
    for j, (kind, col, lab) in enumerate(items):
        e = get(B, kind, "mentions")
        lo, hi = get(B, kind, "mentions", "lower"), get(B, kind, "mentions", "upper")
        c_.barh(len(items) - 1 - j, e, height=0.55, color=col, alpha=0.85, lw=0)
        c_.plot([lo, hi], [len(items) - 1 - j] * 2, color=style.INK, lw=0.8, solid_capstyle="round")
        c_.text(max(hi, e) + 0.04, len(items) - 1 - j, f"{e:.0%}", fontsize=6.0, va="center", color=style.INK)
    c_.set_yticks(range(len(items)))
    c_.set_yticklabels([l for _, _, l in items][::-1], fontsize=6.0)
    c_.grid(axis="y", visible=False)
    c_.set_ylim(-0.6, len(items) - 0.4)
    c_.set_xlim(0, 1.12)
    c_.set_xticks([0, 0.5, 1.0])
    c_.set_xticklabels(["0", "50%", "100%"])
    c_.set_xlabel("reasoning refers to other agents")

    for axx, t in zip(ax, ["a", "b", "c"]):
        axx.set_title(t, loc="left", fontsize=7.2, fontweight="bold", pad=2.5)
    style.finish_paper(fig, ax, str(OUT / f"prewrite_{m}"))
    print(f"[fig] {OUT}/prewrite_{m}.pdf  (best layer {best})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Q9")
