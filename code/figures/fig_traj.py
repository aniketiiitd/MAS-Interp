"""C2 figures: concept drift over 12 turns, revision-locked latent signatures, synchrony, moral drift (frozen tables only)."""
from __future__ import annotations

import argparse
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

style.apply()
GROUP_COLOR = {"social_epistemic": style.TERRACOTTA, "moral": style.PLUM}


def color_of(concept, tax):
    for g, cs in tax["taxonomy"].items():
        if concept in cs:
            return GROUP_COLOR.get(g, style.TEAL)
    return style.SLATE


def pretty(c):
    return c.replace("_", " ")


def drift_over_turns(model, replay, site="dec"):
    tag = f"{model}_T12"
    P = pd.read_parquet(TABLES / f"concept_traj_{model}_{replay}.parquet")
    conf = pd.read_csv(TABLES / f"traj_drift_confirmation_{tag}_{site}.csv")
    tax = json.load(open(ROOT / "data/concepts/taxonomy.json"))
    keep = list(conf[conf.replicates].concept) or list(conf.concept)
    core = {r["id"]: r for r in load_core()}
    P["cluster"] = P.item_id.map(lambda i: core[i]["source_item_id"])
    P["half"] = P.cluster.map(lambda u: "discovery" if h64("disc240091", u) % 2 == 0 else "confirmation")
    H = P[(P.half == "confirmation") & P.cond.isin(["debate", "pool"])]
    fig, ax = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
    for c in keep[:8]:
        g = H.groupby(["turn", "cond"])[f"{site}_{c}"].mean().unstack("cond")
        d = g["debate"] - g["pool"]
        col = color_of(c, tax)
        ax.plot(d.index, d.values, color=col, lw=1.6, alpha=0.9)
        ax.text(d.index[-1] + 0.15, d.values[-1], pretty(c), fontsize=8, va="center", color=style.INK)
    ax.axhline(0, color=style.SLATE, lw=0.7)
    ax.set_xlabel("Debate turn")
    ax.set_ylabel("Debate − independent pool\n(concept z, confirmation half)")
    ax.set_xlim(0.8, 13.5)
    handles = [plt.Line2D([], [], color=c, lw=1.6, label=l) for l, c in (("emotion", style.TEAL), ("social-epistemic", style.TERRACOTTA), ("moral", style.PLUM))]
    ax.legend(handles=handles, loc="upper left", fontsize=8)
    style.finish(fig, ax, FIGS / f"C2_drift_{model}_{site}")


def locked_signatures(model, site="dec"):
    tag = f"{model}_T12"
    L = pd.read_csv(TABLES / f"traj_locked_{tag}_{site}.csv")
    concepts = list(dict.fromkeys(L.concept))[:6]
    fig, axes = plt.subplots(1, len(concepts), figsize=(7.2, 2.3), sharex=True, sharey=True, constrained_layout=True)
    for ax, c in zip(np.atleast_1d(axes), concepts):
        for grp, col, ls in (("revisers", style.TERRACOTTA, "-"), ("dissent_non_revisers", style.SLATE, (0, (1.2, 1.4)))):
            g = L[(L.concept == c) & (L.group == grp)].sort_values("rel_turn")
            ax.fill_between(g.rel_turn, g["mean"] - 1.96 * g["sem"], g["mean"] + 1.96 * g["sem"], color=col, alpha=0.15, lw=0)
            ax.plot(g.rel_turn, g["mean"], color=col, ls=ls, lw=1.5)
        ax.axvline(0, color=style.GRID, lw=1)
        ax.set_title(pretty(c), fontsize=9)
        ax.set_xlabel("Turn relative to revision")
    np.atleast_1d(axes)[0].set_ylabel("Concept z")
    handles = [plt.Line2D([], [], color=style.TERRACOTTA, lw=1.5, label="agents who revise"),
               plt.Line2D([], [], color=style.SLATE, lw=1.5, ls=(0, (1.2, 1.4)), label="faced dissent, did not revise")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, fontsize=8.5)
    style.finish(fig, axes, FIGS / f"C2_locked_{model}_{site}")


def synchrony(model, site="dec"):
    S = pd.read_csv(TABLES / f"traj_synchrony_{model}_T12_{site}.csv")
    piv = S.pivot_table(index="concept", columns="cond", values=["within", "null_95", "null_mean"])
    order = piv[("within", "debate")].sort_values().index
    fig, ax = plt.subplots(figsize=(7.2, 0.22 * len(order) + 0.9), constrained_layout=True)
    y = np.arange(len(order))
    ax.barh(y, piv.loc[order, ("null_95", "debate")], color=style.GRID, height=0.6, label="shuffled-debate 95th pct")
    ax.scatter(piv.loc[order, ("within", "pool")], y, color=style.SLATE, s=18, label="independent pool", zorder=3)
    ax.scatter(piv.loc[order, ("within", "debate")], y, color=style.TERRACOTTA, s=22, label="debate", zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([pretty(c) for c in order], fontsize=8.5)
    ax.set_xlabel("Cross-agent correlation of concept trajectories within a debate")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=8)
    style.finish(fig, ax, FIGS / f"C2_synchrony_{model}_{site}")


def moral(model):
    M = pd.read_csv(TABLES / f"traj_moral_{model}_T12.csv")
    doms = [d for d in DOMAIN_LABEL if d in set(M.domain)]
    concepts = [c for c in dict.fromkeys(M.concept)][:6]
    fig, axes = plt.subplots(1, len(doms), figsize=(7.2, 2.6), sharey=True, constrained_layout=True)
    cmap = plt.get_cmap("Greys")
    for ax, dom in zip(np.atleast_1d(axes), doms):
        for k, c in enumerate(concepts):
            g = M[(M.domain == dom) & (M.concept == c)].sort_values("turn")
            col = style.PLUM if k == 0 else cmap(0.35 + 0.1 * k)
            ax.plot(g.turn, g.drift, color=col, lw=1.4)
            ax.text(g.turn.iloc[-1] + 0.2, g.drift.iloc[-1], pretty(c), fontsize=7, va="center")
        ax.axhline(0, color=style.SLATE, lw=0.7)
        ax.set_title(DOMAIN_LABEL[dom], fontsize=9)
        ax.set_xlabel("Turn")
    np.atleast_1d(axes)[0].set_ylabel("Moral concept z\n(debate − pool)")
    style.finish(fig, axes, FIGS / f"C2_moral_{model}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="+")
    ap.add_argument("--model", default="Q9")
    ap.add_argument("--replay", default="replay_release_600_release_600_T12")
    a = ap.parse_args()
    fns = {"drift": lambda: drift_over_turns(a.model, a.replay), "locked": lambda: locked_signatures(a.model),
           "sync": lambda: synchrony(a.model), "moral": lambda: moral(a.model)}
    for w in a.which:
        fns[w]()
    print("ok")
