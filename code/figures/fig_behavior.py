"""F3 'Where the benchmark changes the answer' + turn trajectories (E2/E4/E5), rendered from frozen tables only."""
from __future__ import annotations

import argparse
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

ENDPOINTS = [("delta_acc", "Accuracy\n(debate − pool)"), ("delta_agree", "Agreement\n(debate − pool)"),
             ("delta_refuse", "Refuse / abstain rate\n(debate − pool)")]


def forest(model, dataset):
    fam = pd.read_csv(TABLES / f"{model}_{dataset}_family.csv")
    fam = fam[fam.level == "family"]
    order = [f for _, f, _ in FAMILY_ORDER]
    dom_of = {f: d for d, f, _ in FAMILY_ORDER}
    fig, axes = plt.subplots(1, len(ENDPOINTS) + 1, figsize=(7.2, 4.0), sharey=True,
                             gridspec_kw=dict(width_ratios=[1, 1, 1, 0.34]), constrained_layout=True)
    y = np.arange(len(order))[::-1]
    for ax, (ep, lab) in zip(axes, ENDPOINTS):
        sub = fam[fam.endpoint == ep].set_index("group")
        ax.axvline(0, color=style.SLATE, lw=0.8, zorder=1)
        for yi, f in zip(y, order):
            if f not in sub.index or not np.isfinite(sub.loc[f, "estimate"]):
                ax.text(0, yi, "–", ha="center", va="center", color=style.MUTED, fontsize=8)
                continue
            r = sub.loc[f]
            c = style.DOMAIN_COLOR[dom_of[f]]
            sig = (r.lower > 0) or (r.upper < 0)
            ax.plot([r.lower, r.upper], [yi, yi], color=c, lw=1.4, solid_capstyle="round", zorder=2)
            ax.scatter([r.estimate], [yi], s=26, color=c if sig else "white", edgecolor=c, linewidth=1.2, zorder=3)
        ax.set_xlabel(lab)
        ax.grid(axis="y", visible=False)
        lim = np.nanmax(np.abs(sub[["lower", "upper"]].to_numpy())) if len(sub) else 0.1
        ax.set_xlim(-max(lim, 0.02) * 1.15, max(lim, 0.02) * 1.15)
    for b in (4.5, 9.5):
        for ax in axes[:-1]:
            ax.axhline(len(order) - 1 - b + 0.0, color=style.GRID, lw=0.8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([FAMILY_LABEL[f] for f in order])
    # right annotation column: n clusters
    ann = axes[-1]
    n = fam[fam.endpoint == "delta_agree"].set_index("group").n_clusters
    for yi, f in zip(y, order):
        ann.text(0.0, yi, f"n={int(n.get(f, 0))}", va="center", ha="left", fontsize=8.5, color=style.SLATE)
    ann.set_axis_off()
    ann.set_xlim(0, 1)
    fig.suptitle(f"Where the benchmark changes the answer ({model}, final turn, 95% cluster-bootstrap CIs; filled = CI excludes 0)",
                 fontsize=9.5, color=style.INK)
    style.finish(fig, axes[:-1], FIGS / f"F3_forest_{model}_{dataset}")


def trajectories(model, dataset):
    ep = pd.read_csv(TABLES / f"{model}_{dataset}_turns.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), constrained_layout=True)
    for ax, (col, lab) in zip(axes, [("agreement", "Public agreement"), ("mean_correct", "Accuracy (target items)"),
                                     ("refuse_rate", "Refuse/abstain rate")]):
        for dom, c in style.DOMAIN_COLOR.items():
            for cond, ls in (("debate", "-"), ("pool", (0, (1.2, 1.4)))):
                g = ep[(ep.domain == dom) & (ep.cond == cond)].groupby("turn")[col].mean()
                if g.notna().sum() == 0:
                    continue
                ax.plot(g.index, g.values, color=c, ls=ls, lw=1.6 if cond == "debate" else 1.2, marker="o" if cond == "debate" else None, ms=3)
        ax.set_xlabel("Turn")
        ax.set_title(lab)
        ax.set_xticks(sorted(ep.turn.unique()))
    handles = [plt.Line2D([], [], color=c, lw=1.6, label=DOMAIN_LABEL[d]) for d, c in style.DOMAIN_COLOR.items()]
    handles += [plt.Line2D([], [], color=style.SLATE, lw=1.6, label="debate"), plt.Line2D([], [], color=style.SLATE, lw=1.2, ls=(0, (1.2, 1.4)), label="independent pool")]
    fig.legend(handles=handles, loc="outside lower center", ncol=5, fontsize=8.5)
    style.finish(fig, axes, FIGS / f"S_trajectories_{model}_{dataset}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Q9")
    ap.add_argument("--dataset", default="core")
    a = ap.parse_args()
    forest(a.model, a.dataset)
    trajectories(a.model, a.dataset)
    print("ok")
