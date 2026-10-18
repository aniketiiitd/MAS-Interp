"""F2 consensus-dispersion phase portrait, F7 who-influences-whom, F9 recourse, F10 monitor risk-coverage."""
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


def f2_phase(model="Q9", replay="replay_core_release_600"):
    D = pd.read_csv(TABLES / f"dispersion_{model}_{replay}.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), sharex=True, sharey=True, constrained_layout=True)
    ylo, yhi = D.dispersion_resid.quantile([0.02, 0.98])
    pool_med = D[D.cond == "pool"].dispersion_resid.median()
    for ax, (dom, c) in zip(axes, style.DOMAIN_COLOR.items()):
        g = D[D.domain == dom]
        for cond, ls, lw in (("pool", (0, (1.2, 1.4)), 1.3), ("debate", "-", 1.8)):
            m = g[g.cond == cond].groupby("turn")[["agreement", "dispersion_resid"]].median()
            # faint individual trajectories (debate only, subsample 40 items)
            if cond == "debate":
                ids = sorted(g.item_id.unique(), key=lambda i: h64("f2", i))[:40]
                for iid in ids:
                    t = g[(g.item_id == iid) & (g.cond == cond)].sort_values("turn")
                    ax.plot(t.agreement + (h64("jx", iid) % 100 - 50) / 2000, t.dispersion_resid, color=c, alpha=0.08, lw=0.7)
            ax.plot(m.agreement, m.dispersion_resid, color=c if cond == "debate" else style.SLATE, ls=ls, lw=lw, zorder=3)
            ax.annotate("", xy=(m.agreement.iloc[-1], m.dispersion_resid.iloc[-1]), xytext=(m.agreement.iloc[-2], m.dispersion_resid.iloc[-2]),
                        arrowprops=dict(arrowstyle="-|>", color=c if cond == "debate" else style.SLATE, lw=lw), zorder=4)
            ax.scatter(m.agreement.iloc[:1], m.dispersion_resid.iloc[:1], s=16, color="white", edgecolor=style.INK, zorder=5, lw=0.8)
        ax.axhline(pool_med, color=style.GRID, lw=1.0, zorder=0)
        ax.set_title(DOMAIN_LABEL[dom])
        ax.set_xlabel("Public agreement")
        ax.set_ylim(ylo, yhi)
    axes[0].set_ylabel("Residualized private-state\ndispersion")
    handles = [plt.Line2D([], [], color=style.INK, lw=1.8, label="debate (domain median)"),
               plt.Line2D([], [], color=style.SLATE, lw=1.3, ls=(0, (1.2, 1.4)), label="independent pool"),
               plt.Line2D([], [], color=style.INK, lw=0, marker="o", mfc="white", label="turn 0")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=8.5)
    style.finish(fig, axes, FIGS / f"F2_phase_{model}")


def f7_edges():
    T = pd.read_csv(TABLES / "edges_influence.csv")
    full = T[T.component == "full"].copy()
    full["excess"] = full.estimate - full.self_baseline
    order_models = ["Q4", "Q9", "Q27", "Q35", "G4", "G12", "G26", "G31", "L31", "M7", "O7", "P4"]
    rows = full[full.receiver != full.sender].copy()
    rows = rows[rows.receiver.isin(["Q9", "G4", "Q4"]) | rows.sender.isin(["Q9"])]
    rows["label"] = rows.receiver + " ← " + rows.sender
    rows["key"] = rows.receiver.map(order_models.index) * 100 + rows.sender.map(order_models.index)
    rows = rows.sort_values("key")
    comp = T.set_index(["receiver", "sender", "component"]).estimate
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.2, 0.18 * len(rows) + 1.2), sharey=True, constrained_layout=True,
                                  gridspec_kw=dict(width_ratios=[1.2, 1]))
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows.itertuples()):
        moe = MODELS[r.sender]["arch"] == "moe"
        ax.plot([r.lower, r.upper], [yi, yi], color=style.SLATE, lw=1.1)
        ax.scatter([r.estimate], [yi], marker="s" if moe else "o", s=22, color=style.INK, zorder=3)
        ax.scatter([r.self_baseline], [yi], marker="|", s=40, color=style.MUTED, zorder=2)
        c = comp.get((r.receiver, r.sender, "content"), np.nan)
        p = comp.get((r.receiver, r.sender, "provenance"), np.nan)
        ax2.scatter([c], [yi + 0.12], color=style.TEAL, s=18, marker="o")
        ax2.scatter([p], [yi - 0.12], color=style.TERRACOTTA, s=18, marker="D")
    ax.set_yticks(y)
    ax.set_yticklabels(rows.label)
    ax.set_xlabel("P(receiver adopts sender verdict)")
    ax2.axvline(0, color=style.SLATE, lw=0.7)
    ax2.set_xlabel("Component of adoption")
    for a in (ax, ax2):
        a.grid(axis="y", visible=False)
    handles = [plt.Line2D([], [], color=style.TEAL, marker="o", lw=0, label="content (full − verdict-only)"),
               plt.Line2D([], [], color=style.TERRACOTTA, marker="D", lw=0, label="provenance (expert − anon)"),
               plt.Line2D([], [], color=style.INK, marker="s", lw=0, label="MoE sender"),
               plt.Line2D([], [], color=style.MUTED, marker="|", lw=0, markersize=9, label="self-edge baseline")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=8)
    style.finish(fig, [ax, ax2], FIGS / "F7_edges")


def f9_recourse(model="Q9"):
    T = pd.read_csv(TABLES / f"recourse_{model}.csv")
    routes = [("a_neutral_message", "replace message (neutral)"), ("b_source_label_removed", "remove source label"),
              ("c_provenance_state_patch", "patch source-token state"), ("d_provenance_steer_-2", "steer provenance −2 SD"),
              ("e_evidence_steer_-2", "steer evidence −2 SD"), ("anon_reference", "anonymous source (reference)")]
    fig, ax = plt.subplots(figsize=(7.2, 2.7), constrained_layout=True)
    y = np.arange(len(routes))[::-1]
    for yi, (r, lab) in zip(y, routes):
        for scen, c, dy, mk in (("harmful", style.TERRACOTTA, 0.14, "o"), ("side_effect", style.TEAL, -0.14, "s")):
            s = T[(T.route == r) & (T.scenario == scen)]
            if not len(s):
                continue
            s = s.iloc[0]
            ax.plot([s.lower, s.upper], [yi + dy, yi + dy], color=c, lw=1.4)
            ax.scatter([s.d_p_correct], [yi + dy], color=c, marker=mk, s=24, zorder=3)
    ax.axvline(0, color=style.SLATE, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, lab in routes])
    ax.set_xlabel("Change in P(correct option) relative to the unrepaired expert message")
    ax.grid(axis="y", visible=False)
    handles = [plt.Line2D([], [], color=style.TERRACOTTA, marker="o", lw=1.4, label="repair: expert pushes the wrong option"),
               plt.Line2D([], [], color=style.TEAL, marker="s", lw=1.4, label="side effect: expert pushes the right option")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, fontsize=8.5)
    style.finish(fig, ax, FIGS / f"F9_recourse_{model}")


def f10_monitor(model="Q9"):
    C = pd.read_csv(TABLES / f"monitor_curves_{model}.csv")
    R = pd.read_csv(TABLES / f"monitor_{model}.csv").set_index("monitor")
    fig, ax = plt.subplots(figsize=(7.2, 2.9), constrained_layout=True)
    styles = {"public": (style.SLATE, (0, (1.2, 1.4))), "transcript": (style.MUTED, (0, (4, 1.5))), "activation": (style.TEAL, "-"),
              "cpt": (style.PLUM, "-"), "cpt+public": (style.TERRACOTTA, "-")}
    for m, (c, ls) in styles.items():
        g = C[C.monitor == m]
        if not len(g):
            continue
        ax.plot(g.coverage, g.accuracy, color=c, ls=ls, lw=1.6)
        ax.text(1.01, g.accuracy.iloc[-1] + 0.0, f"{m}  AURC {R.loc[m, 'aurc']:.3f}", color=style.INK, fontsize=8, va="center",
                transform=ax.get_yaxis_transform())
    ax.axvline(0.8, color=style.GRID, lw=1)
    ax.set_xlabel("Fraction of episodes accepted (lowest predicted risk first)")
    ax.set_ylabel("Accuracy of accepted\nfinal consensus")
    ax.set_xlim(0.05, 1.0)
    style.finish(fig, ax, FIGS / f"F10_monitor_{model}")


def f_b2(model="Q9", dataset="release_600"):
    """B2: consensus lock-in and private defection."""
    import json as _json
    S = pd.read_csv(TABLES / f"long_{model}_{dataset}_summary.csv")
    J = _json.load(open(TABLES / f"long_{model}_{dataset}_private.json"))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), constrained_layout=True, gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
    for ax, (ep, lab) in zip(axes[:2], [("unanimous", "Unanimous groups"), ("wrong_unanimous", "Unanimous and wrong")]):
        for cond, col, ls in (("debate", style.TERRACOTTA, "-"), ("pool", style.SLATE, (0, (1.2, 1.4)))):
            g = S[(S.cond == cond) & (S.endpoint == ep)].sort_values("turn")
            ax.fill_between(g.turn, g.lower, g.upper, color=col, alpha=0.15, lw=0)
            ax.plot(g.turn, g.estimate, color=col, ls=ls, lw=1.7)
            ax.text(g.turn.iloc[-1] + 0.3, g.estimate.iloc[-1], cond, fontsize=8, va="center", color=style.INK)
        ax.set_title(lab, fontsize=9.5)
        ax.set_xlabel("Turn")
        ax.set_xlim(-0.3, 13.5)
    ax = axes[2]
    rows = [("pool, publicly unanimous", J["pool_public_unanimous"]["private_switch"], J["pool_public_unanimous"]["lower"], J["pool_public_unanimous"]["upper"], style.SLATE),
            ("debate, publicly unanimous", J["debate_public_unanimous"]["private_switch"], J["debate_public_unanimous"]["lower"], J["debate_public_unanimous"]["upper"], style.TERRACOTTA),
            ("paired: unanimous in both", J["unanimous_both_paired_diff"]["estimate"], J["unanimous_both_paired_diff"]["lower"], J["unanimous_both_paired_diff"]["upper"], style.INK),
            ("paired: + same answer", J["unanimous_both_same_answer_paired_diff"]["estimate"], J["unanimous_both_same_answer_paired_diff"]["lower"], J["unanimous_both_same_answer_paired_diff"]["upper"], style.INK)]
    y = np.array([3.4, 2.4, 1.0, 0.0])
    for yi, (lab, e, lo, hi, col) in zip(y, rows):
        ax.plot([lo, hi], [yi, yi], color=col, lw=1.5)
        ax.scatter([e], [yi], color=col, s=26, zorder=3, marker="o" if yi > 2 else "D")
    ax.axvline(0, color=style.GRID, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlabel("Private switch rate / debate − pool")
    ax.set_title("Private defection after consensus", fontsize=9.5)
    ax.grid(axis="y", visible=False)
    style.finish(fig, axes, FIGS / f"B2_lockin_private_{model}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="+")
    a = ap.parse_args()
    for w in a.which:
        {"f2": f2_phase, "f7": f7_edges, "f9": f9_recourse, "f10": f10_monitor, "b2": f_b2}[w]()
    print("ok")
