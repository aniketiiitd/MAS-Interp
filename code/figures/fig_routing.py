"""Where a peer's influence reaches the decision, in three architectures (reads outputs/tables/lit_routing_*.csv).

Panel a  removed share against relative depth, one line per model, with the matched own-response control.
Panel b  cumulative removed share: attention edges blocked at every attention layer up to that depth.
Panel c  what attention can remove at all, per model, against the text-level deletion of the same message.
Panel d  Gemma-4: recovery of a source-descriptor interchange through the residual stream alone and with the per-layer
         embedding channel, compared with the same residual-only patch in the two models that have no such channel.
Usage: python figures/fig_routing.py
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
MNAME = {"Q9": "Qwen3.5-9B", "L31": "Llama-3.1-8B", "G4": "Gemma-4-E4B", "Q08": "Qwen3.5-0.8B"}
MCOL = {"Q9": style.TEAL, "L31": style.SLATE, "G4": style.PLUM, "Q08": style.TEAL}
ARCH = {"Q9": "8 of 32 attention", "L31": "32 of 32 attention", "G4": "7 of 42 global", "Q08": "6 of 24 attention"}
ORDER = ["L31", "Q9", "G4", "Q08"]


def band(ax, x, e, lo, hi, c, label, ls="-", marker="o"):
    ax.plot(x, e, color=c, ls=ls, marker=marker, label=label, zorder=3)
    ax.fill_between(x, lo, hi, color=c, alpha=0.13, lw=0, zorder=2)


def main():
    L = pd.read_csv(TABLES / "lit_routing_layers.csv")
    S = pd.read_csv(TABLES / "lit_routing_models.csv")
    L = L[L.domain == "all"]
    models = [m for m in ORDER if m in set(L.model)]
    sp_path = TABLES / "lit_routing_span.csv"
    SP = pd.read_csv(sp_path) if sp_path.exists() else None
    fig, ax = style.strip(4, height=2.0, width_ratios=[1.0, 1.0, 1.3, 1.1])

    for k, (kind, a, ttl) in enumerate([("peer_layer", ax[0], "one layer"),
                                        ("peer_cumulative", ax[1], "cumulative")]):
        for m in models:
            g = L[(L.model == m) & (L.kind == kind)].sort_values("depth")
            if len(g):
                band(a, g.depth, g.share, g.share_lo, g.share_hi, MCOL[m], f"{MNAME[m]} ({ARCH[m]})")
        if k == 0:
            for m in models:
                g = L[(L.model == m) & (L.kind == "own_layer")].sort_values("depth")
                if len(g):
                    a.plot(g.depth, g.share, color=MCOL[m], ls=":", lw=0.9, marker=None, alpha=0.8, zorder=1)
        a.axhline(0, color=style.MUTED, lw=0.5, zorder=1)
        a.axhline(1, color=style.MUTED, lw=0.5, ls=":", zorder=1)
        a.set_xlabel("relative depth")
        a.set_title(ttl)
        a.set_xlim(-0.03, 1.03)
        v = L[(L.model.isin(models)) & (L.kind.isin([kind, "own_layer"]))].share.to_numpy(float)
        v = v[np.isfinite(v)]
        if len(v):
            lo_, hi_ = float(np.nanmin(v)), float(np.nanmax(v))
            a.set_ylim(min(-0.12, lo_ - 0.08), max(1.12, hi_ + 0.08))
    ax[0].set_ylabel("share of the peer's\ninfluence removed")
    ax[0].plot([], [], color=style.MUTED, ls=":", lw=0.9, label="own response, matched length")
    ax[0].legend(loc="upper left", frameon=False, handlelength=1.1, fontsize=5.2, labelspacing=0.25)

    bars = [("peer_all", "decision"), ("peer_allq_all", "all later"), ("own_all", "own span"), ("qctrl_all", "other query")]
    NAME = {"peer_all": "peer keys blocked at every attention layer (decision query)",
            "peer_allq_all": "peer keys blocked for every later query at every attention layer",
            "own_all": "control: own-response keys of matched length",
            "qctrl_all": "control: peer keys blocked for the prefill tokens, not the decision token"}
    w, xs = 0.8 / max(len(models), 1), np.arange(len(bars))
    for j, m in enumerate(models):
        g = S[S.model == m].set_index("arm")
        e = [g.share.get(NAME[b], np.nan) for b, _ in bars]
        lo = [g.share_lo.get(NAME[b], np.nan) for b, _ in bars]
        hi = [g.share_hi.get(NAME[b], np.nan) for b, _ in bars]
        x = xs + (j - (len(models) - 1) / 2) * w
        e, lo, hi = np.array(e, float), np.clip(lo, -0.3, 1.3), np.clip(hi, -0.3, 1.3)
        ax[2].bar(x, e, w * 0.92, color=MCOL[m], zorder=3)
        ax[2].errorbar(x, e, yerr=[np.maximum(e - lo, 0), np.maximum(hi - e, 0)], fmt="none",
                       ecolor=style.SLATE, elinewidth=0.6, capsize=1.2, zorder=4)
    ax[2].axhline(1.0, color=style.TERRACOTTA, lw=0.7, ls="--", zorder=2)
    ax[2].text(len(bars) - 0.55, 1.02, "deleting the message", color=style.TERRACOTTA, fontsize=5.6, ha="right", va="bottom")
    ax[2].axhline(0, color=style.MUTED, lw=0.5, zorder=1)
    ax[2].set_xticks(xs)
    ax[2].set_xticklabels([t for _, t in bars], fontsize=5.6, rotation=22, ha="right", rotation_mode="anchor")
    ax[2].set_xlabel("query that may not read the peer")
    ax[2].set_ylabel("share removed")
    ax[2].set_title("all layers")
    ax[2].set_ylim(-0.35, 1.35)

    if SP is not None and len(SP):
        KC = {"residual only": style.SLATE, "residual + per-layer embeddings": style.PLUM,
              "per-layer embeddings only": style.TERRACOTTA}
        for m in [x for x in models if x in set(SP.model)]:
            g0 = SP[SP.model == m]
            for kind, c in KC.items():
                g = g0[g0.kind == kind].sort_values("layer")
                if not len(g):
                    continue
                lbl = kind if m == "G4" else f"{MNAME[m]}, residual"
                ax[3].plot(g.layer / g.layer.max(), g.recovery, color=(c if m == "G4" else MCOL[m]),
                           ls=("-" if m == "G4" else "--"), lw=(1.2 if m == "G4" else 0.9),
                           marker=("o" if m == "G4" else None), label=lbl, zorder=3)
        ax[3].axhline(0, color=style.MUTED, lw=0.5)
        ax[3].axhline(1, color=style.MUTED, lw=0.5, ls=":")
        ax[3].set_xlabel("relative depth")
        ax[3].set_ylabel("source contrast transported")
        ax[3].set_title("span interchange")
        ax[3].set_ylim(-0.25, 1.15)
        ax[3].legend(loc="upper right", frameon=False, fontsize=5.2, labelspacing=0.25, handlelength=1.1)
    for a in ax[:2]:
        a.grid(axis="y", alpha=0.25)
    ax[2].grid(axis="y", alpha=0.25)
    style.finish_paper(fig, ax, str(OUT / "fig_routing"))
    print("wrote", OUT / "fig_routing.pdf")


if __name__ == "__main__":
    main()
