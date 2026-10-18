"""Figures: a decision state taken out of one model, carried into another model's space, and written into its forward pass.

Usage: python figures/fig_latent_cross.py Q9toL31 Q9toG4 G4toQ9
Pairs whose tables are missing are skipped, so the script can run while the queue is still working through the pairs.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "figures"))
from common import *  # noqa
import style
from style import plt
from matplotlib.lines import Line2D

style.apply_paper()
OUT = ROOT / "outputs/figures/paper"
OUT.mkdir(parents=True, exist_ok=True)
MNAME = {"Q9": "Qwen3.5-9B", "L31": "Llama-3.1-8B", "G4": "Gemma-4-E4B", "Q08": "Qwen3.5-0.8B"}
COLORS = [style.TERRACOTTA, style.TEAL, style.PLUM]
S_, M_ = style.SLATE, style.MUTED
CTRL = [("mapped_ridge", "the sender's state, carried"),
        ("within_model", "the receiver's own state"),
        ("mapped_proc", "carried by a rotation only"),
        ("mapped_shuffled_fit", "map fitted on shuffled pairs"),
        ("mapped_same_letter_other_item", "another item, same letter"),
        ("mapped_deranged", "another episode's sender"),
        ("mapped_mean_sender", "the average sender state"),
        ("raw_unmapped", "the raw state, not carried"),
        ("rotated", "carried, then scrambled"),
        ("random_vector", "random vector, same size"),
        ("other_position", "carried, written elsewhere")]


def pair_label(p):
    s, r = p.split("to")
    return f"{MNAME.get(s, s)} → {MNAME.get(r, r)}"


def load(pairs):
    out = []
    for p in pairs:
        f = TABLES / f"lit_latentx_{p}_transfer.csv"
        if not f.exists():
            print(f"[skip] {p}: no tables yet")
            continue
        d = dict(pair=p, T=pd.read_csv(f), K=pd.read_csv(TABLES / f"lit_latentx_{p}_rank.csv"),
                 M=pd.read_csv(TABLES / f"lit_latentx_{p}_map.csv"),
                 S=json.load(open(TABLES / f"lit_latentx_{p}_summary.json")))
        for name in ("fidelity", "truth"):
            q = TABLES / f"lit_latentx_{p}_{name}.csv"
            d[name] = pd.read_csv(q) if q.exists() else pd.DataFrame()
        out.append(d)
    return out


def band(ax, x, y, lo, hi, c, label=None, ls="-", marker="o"):
    ax.plot(x, y, color=c, ls=ls, marker=marker, ms=2.6, label=label)
    ax.fill_between(x, lo, hi, color=c, alpha=0.15, lw=0)


def hero(data, stem="fig_latent_cross"):
    fig, ax = style.strip(3, 1.85, [1.0, 1.05, 1.05])
    # a: how well the state survives the change of space
    for i, d in enumerate(data):
        m = d["M"]
        g = m[(m.split == "heldout") & (m["map"] == "ridge") & (m.f_sender == m.f_recv)].sort_values("f_sender")
        ax[0].plot(g.f_sender, g.top1, color=COLORS[i], marker="o", ms=2.8, label=pair_label(d["pair"]))
        gp = m[(m.split == "heldout") & (m["map"] == "proc") & (m.f_sender == m.f_recv)].sort_values("f_sender")
        ax[0].plot(gp.f_sender, gp.top1, color=COLORS[i], ls="--", marker="^", ms=2.4, lw=0.9)
        gs = m[(m.split == "heldout") & (m["map"] == "shuffled")]
        if len(gs):
            ax[0].scatter(gs.f_sender, gs.top1, color=M_, marker="x", s=10, zorder=3)
    ax[0].axhline(0, color=S_, lw=0.7)
    ax[0].set_ylim(-0.03, 1.03)
    ax[0].set_xlabel("relative depth")
    ax[0].set_ylabel("matched state ranked first")
    ax[0].set_title("a  Does the state survive the trip?", loc="left")
    h, _ = ax[0].get_legend_handles_labels()
    ax[0].legend(handles=h + [Line2D([], [], color=S_, ls="--", marker="^", ms=2.4, lw=0.9, label="a rotation only"),
                              Line2D([], [], color=M_, ls="", marker="x", ms=3, label="fitted on shuffled pairs")],
                 loc="lower left", fontsize=5.2, labelspacing=0.22)
    # b: what it does to the decision, against the within-model ceiling
    for i, d in enumerate(data):
        T = d["T"]
        g = T[T.arm == "mapped_ridge"].sort_values("depth")
        band(ax[1], g.depth, g.transfer, g.lower, g.upper, COLORS[i], pair_label(d["pair"]))
        w = T[T.arm == "within_model"].sort_values("depth")
        ax[1].plot(w.depth, w.transfer, color=COLORS[i], ls="--", lw=0.9, marker="^", ms=2.2)
    ax[1].axhline(1, color=S_, lw=0.7, ls=":")
    ax[1].axhline(0, color=S_, lw=0.7)
    ax[1].set_xlabel("relative depth of the transplant")
    ax[1].set_ylabel("share of the written response")
    ax[1].set_title("b  Does it move the decision?", loc="left")
    h, l = ax[1].get_legend_handles_labels()
    ax[1].legend(handles=h + [Line2D([], [], color=S_, ls="--", marker="^", ms=2.2, lw=0.9, label="the receiver's own state")],
                 loc="lower left", fontsize=5.2, labelspacing=0.22)
    # c: how many directions it takes, cross-model against within-model
    for i, d in enumerate(data):
        T, f = d["T"], d["S"]["primary_depth"]
        for arm, ls, mk in (("rank_cross", "-", "o"), ("rank_within", "--", "^")):
            g = T[(T.arm == arm) & (T.depth == f) & (T.basis == "pca")].sort_values("k")
            if not len(g):
                continue
            band(ax[2], g.k, g.transfer, g.lower, g.upper, COLORS[i], pair_label(d["pair"]) if arm == "rank_cross" else None,
                 ls=ls, marker=mk)
        for _, r in d["K"][(d["K"].depth == f) & d["K"].k_50.notna()].iterrows():
            ax[2].axvline(r.k_50, color=COLORS[i], lw=0.6, ls=":" if r.arm == "rank_cross" else (0, (1, 2)))
    ax[2].axhline(0, color=S_, lw=0.7)
    ax[2].set_xscale("log", base=2)
    ax[2].set_xlabel("number of directions written")
    ax[2].set_ylabel("share of the written response")
    ax[2].set_title("c  How many directions?", loc="left")
    h, l = ax[2].get_legend_handles_labels()
    ax[2].legend(handles=h + [Line2D([], [], color=S_, ls="--", marker="^", ms=2.2, lw=0.9, label="within-model")],
                 loc="upper left", fontsize=5.2, labelspacing=0.22)
    style.finish_paper(fig, ax, OUT / stem)


def controls(data, stem="fig_latent_cross_controls"):
    fig, ax = style.strip(3, 2.0, [1.5, 0.72, 1.0])
    # a: every arm at the primary depth
    names = [c for c in CTRL]
    for i, d in enumerate(data):
        T, f = d["T"], d["S"]["primary_depth"]
        for yi, (arm, lab) in enumerate(names):
            r = T[(T.arm == arm) & (T.depth == f)]
            if not len(r):
                continue
            r = r.iloc[0]
            y = len(names) - yi + (i - (len(data) - 1) / 2) * 0.26
            ax[0].plot([r.lower, r.upper], [y, y], color=COLORS[i], lw=1.0)
            ax[0].scatter([r.transfer], [y], s=11, color=COLORS[i], zorder=3)
    ax[0].axvline(0, color=S_, lw=0.7)
    ax[0].axvline(1, color=S_, lw=0.7, ls=":")
    ax[0].set_yticks([len(names) - yi for yi in range(len(names))])
    ax[0].set_yticklabels([lab for _, lab in names], fontsize=5.0)
    ax[0].grid(axis="y", visible=False)
    ax[0].set_xlabel("share of the written response")
    ax[0].set_title("a  What the state has to be", loc="left")
    for i, d in enumerate(data):
        ax[0].text(0.98, 0.02 + 0.075 * (len(data) - 1 - i), pair_label(d["pair"]), transform=ax[0].transAxes,
                   ha="right", va="bottom", fontsize=5.2, color=COLORS[i])
    # b: transfer against how well the state was carried
    for i, d in enumerate(data):
        F = d["fidelity"]
        F = F[F.tercile.isin(["low", "mid", "high"])] if len(F) else F
        if not len(F):
            continue
        x = np.arange(len(F)) + (i - (len(data) - 1) / 2) * 0.16
        ax[1].errorbar(x, F.transfer, yerr=[F.transfer - F.lower, F.upper - F.transfer], color=COLORS[i], marker="o",
                       ms=3, lw=0, elinewidth=0.9, capsize=1.5)
        ax[1].set_xticks(np.arange(len(F)))
        ax[1].set_xticklabels([f"{c:.2f}" for c in F.cos_mean], fontsize=5.6)
    ax[1].axhline(0, color=S_, lw=0.7)
    ax[1].set_xlabel("how well it was carried")
    ax[1].set_ylabel("share of the written response")
    ax[1].set_title("b  Carried better, more effect", loc="left")
    # c: the answer or the truth
    ys, seen = [], 0
    for i, d in enumerate(data):
        TR = d["truth"]
        if not len(TR):
            continue
        for correct in (True, False):
            r = TR[(TR.channel == "mapped_ridge") & (TR.sender_correct == correct)]
            if not len(r):
                continue
            r = r.iloc[0]
            y = -seen
            seen += 1
            ys.append((y, f"into {MNAME.get(d['pair'].split('to')[1], '')}, sender {'right' if correct else 'wrong'}"))
            for val, lo, hi, c, mk, off in ((r.toward_sender, r.ts_lower, r.ts_upper, COLORS[i], "o", 0.15),
                                            (r.toward_target, r.tt_lower, r.tt_upper, S_, "s", -0.15)):
                ax[2].plot([lo, hi], [y + off, y + off], color=c, lw=1.0)
                ax[2].scatter([val], [y + off], s=12, color=c, marker=mk, zorder=3)
    ax[2].axvline(0, color=S_, lw=0.7)
    ax[2].set_yticks([y for y, _ in ys])
    ax[2].set_yticklabels([l for _, l in ys], fontsize=5.4)
    ax[2].grid(axis="y", visible=False)
    ax[2].set_xlabel("margin shift (logits)")
    ax[2].set_title("c  Answer or truth?", loc="left")
    ax[2].set_ylim(min(y for y, _ in ys) - 0.55, 0.95)
    ax[2].legend(handles=[Line2D([], [], marker="o", ls="", color=COLORS[0], ms=3, label="toward the sender's answer"),
                          Line2D([], [], marker="s", ls="", color=S_, ms=3, label="toward the correct answer")],
                 loc="upper left", fontsize=5.0)
    style.finish_paper(fig, ax, OUT / stem)


def depth_grid(data, stem="fig_latent_cross_depths"):
    """Retrieval accuracy for every (sender depth, receiver depth) pair: is the match best at equal relative depth?"""
    n = len(data)
    fig, ax = style.strip(max(n, 2), 1.7)
    for i, d in enumerate(data):
        m = d["M"]
        g = m[(m.split == "heldout") & (m["map"] == "ridge")]
        piv = g.pivot_table(index="f_recv", columns="f_sender", values="top1")
        im = ax[i].imshow(piv.to_numpy(), cmap=style.SEQUENTIAL, vmin=0, vmax=1, origin="lower", aspect="auto")
        ax[i].set_xticks(range(len(piv.columns)), [f"{c:.2f}" for c in piv.columns], fontsize=5.4)
        ax[i].set_yticks(range(len(piv.index)), [f"{c:.2f}" for c in piv.index], fontsize=5.4)
        ax[i].grid(False)
        ax[i].set_xlabel("sender depth")
        if i == 0:
            ax[i].set_ylabel("receiver depth")
        ax[i].set_title(pair_label(d["pair"]), loc="left", fontsize=6.4)
        for yi in range(len(piv.index)):
            for xi in range(len(piv.columns)):
                v = piv.to_numpy()[yi, xi]
                if np.isfinite(v):
                    ax[i].text(xi, yi, f"{v:.2f}", ha="center", va="center", fontsize=4.6,
                               color="white" if v > 0.55 else style.INK)
    for j in range(len(data), len(ax)):
        ax[j].axis("off")
    fig.colorbar(im, ax=ax[len(data) - 1], fraction=0.045, pad=0.02).set_label("matched state ranked first", fontsize=5.6)
    style.finish_paper(fig, ax, OUT / stem)


if __name__ == "__main__":
    pairs = sys.argv[1:] or ["Q9toL31", "Q9toG4", "G4toQ9"]
    data = load(pairs)
    if not data:
        sys.exit("no pairs available yet")
    hero(data)
    controls(data)
    depth_grid(data)
    print(f"[done] wrote {OUT}/fig_latent_cross{{,_controls,_depths}}.pdf for {[d['pair'] for d in data]}")
