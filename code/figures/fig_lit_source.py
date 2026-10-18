"""Hero 1 figure: who said it — authority is read out as reliability (analysis/lit_source.py tables).
Usage: python figures/fig_lit_source.py Q9 [L31 G4]
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
MNAME = {"Q9": "Qwen3.5-9B", "L31": "Llama-3.1-8B-Instruct", "G4": "Gemma-4-E4B"}
MCOL = {"Q9": style.PLUM, "L31": style.TEAL, "G4": style.TERRACOTTA}


def hbar(ax, y, e, lo, hi, c, m="o", s=30, filled=True, label=None):
    ax.plot([lo, hi], [y, y], color=c, lw=1.3, solid_capstyle="round", zorder=2)
    ax.scatter([e], [y], s=s, marker=m, facecolor=c if filled else "white", edgecolor=c, lw=1.2, zorder=3, label=label)


def main(models):
    fig = plt.figure(figsize=(7.2, 5.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05])
    # a: 2 x 3 cells (primary model)
    m0 = models[0]
    C = pd.read_csv(TABLES / f"lit_source_{m0}_cells.csv").set_index("source")
    ax = fig.add_subplot(gs[0, 0])
    recs = [("good", "strong track record"), ("unst", "unknown track record"), ("bad", "poor track record")]
    for si, (s, lab, c, mk) in enumerate((("prof", "tenured professor", style.PLUM, "o"), ("ord", "ordinary member of the public", style.TEAL, "s"))):
        for ri, (r, _) in enumerate(recs):
            row = C.loc[f"{s}_{r}"]
            hbar(ax, 2 - ri + (0.13 if si == 0 else -0.13), row.est, row.lower, row.upper, c, mk, label=lab if ri == 0 else None)
    ax.axvline(0, color=style.SLATE, lw=0.8)
    ax.set_yticks(range(3))
    ax.set_yticklabels([l for _, l in recs][::-1])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Shift toward the other agent's answer\nvs. the same message without a label (logits)")
    ax.set_title("a  Same message, different source", loc="left")
    ax.legend(loc="upper left", fontsize=7.5)
    # b: represented vs read out (primary model): norm ratio of the status and track-record codes vs their weight in the decision readout
    ax = fig.add_subplot(gs[0, 1])
    nl = {"Q9": 32, "L31": 32, "G4": 42}[m0]
    RA = TABLES / f"lit_source_{m0}_readout_alignment.csv"
    E = pd.read_csv(TABLES / f"lit_source_{m0}_effects.csv").set_index("effect")
    b = E.loc["status share (behaviour)"]
    if RA.exists():
        R = pd.read_csv(RA)
        R["depth"] = R.layer / (nl - 1)
        R = R[R.layer >= 9]
        rep = R.norm_ratio_status_over_record / (1 + R.norm_ratio_status_over_record)
        ax.plot(R.depth, rep, color=style.PLUM, lw=1.5, marker="o", ms=3.5, label="represented (size of the status code)")
        ax.plot(R.depth, R.readout_share_status, color=style.TEAL, lw=1.5, marker="s", ms=3.5, label="read out by the decision")
    ax.axhspan(b.lower, b.upper, color=style.TERRACOTTA, alpha=0.18, lw=0)
    ax.axhline(b.est, color=style.TERRACOTTA, lw=1.1, ls="--", label="behaviour")
    ax.axhline(0.5, color=style.SLATE, lw=0.6, ls=":")
    ax.set_ylim(0, 0.75)
    ax.set_xlim(0.25, 1.02)
    ax.set_xlabel("Relative depth (layer / last layer)")
    ax.set_ylabel("Status share (status vs track record)")
    ax.set_title("b  Status is encoded, track record is used", loc="left")
    ax.legend(loc="upper right", fontsize=7.2)
    # c/d: interchange interventions at the decision token (primary model)
    S = pd.read_csv(TABLES / f"lit_source_{m0}_swap.csv")
    S["depth"] = S.layer / (nl - 1)
    for k, (contrast, title) in enumerate((("status", "c  Swap professor into ordinary member"),
                                           ("record", "d  Swap strong into poor track record"))):
        ax = fig.add_subplot(gs[1, k])
        for kind, lab, c, ls in (("full", "whole decision-token state", style.INK, "-"),
                                 ("proj_rec", "one track-record direction", style.TEAL, "--"),
                                 ("proj_stat", "one status direction", style.PLUM, "--"),
                                 ("proj_rand", "one random direction", style.MUTED, ":")):
            g = S[(S.contrast == contrast) & (S.kind == kind)].sort_values("depth")
            ax.plot(g.depth, g.recovery, color=c, lw=1.4, ls=ls, marker="o", ms=3, label=lab)
            ax.fill_between(g.depth, g.lower, g.upper, color=c, alpha=0.15, lw=0)
        ax.axhline(0, color=style.SLATE, lw=0.7)
        ax.axhline(1, color=style.SLATE, lw=0.5, ls=":")
        ax.set_xlim(0, 1.02)
        ax.set_xlabel("Relative depth of the swap")
        ax.set_ylabel("Share of the behavioural effect transferred")
        ax.set_title(title, loc="left")
        if k == 0:
            ax.legend(loc="center left", fontsize=7.2)
    style.finish(fig, fig.axes, ROOT / "outputs/figures" / f"H1_source_reliability_{'_'.join(models)}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["Q9"])
