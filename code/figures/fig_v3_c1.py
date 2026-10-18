"""V3-C1 CPU robustness figures from outputs/tables/c1r_<M>_*.csv (analysis/c1_robust.py)."""
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
DOMS = ["knowledge_reasoning", "values_moral_judgment", "safety_refusal"]
DLAB = {"ALL": "All items", "knowledge_reasoning": "Knowledge", "values_moral_judgment": "Values", "safety_refusal": "Safety"}


def ci_h(ax, y, est, lo, hi, color, marker="o", filled=True, size=26, lw=1.2, label=None):
    ax.plot([lo, hi], [y, y], color=color, lw=lw, solid_capstyle="round", zorder=2)
    ax.scatter([est], [y], marker=marker, s=size, facecolor=color if filled else "white", edgecolor=color, lw=1.1, zorder=3,
               label=label)


def fig_iqp(model):
    R = pd.read_csv(TABLES / f"c1r_{model}_iqp_adjustments.csv")
    Sc = pd.read_csv(TABLES / f"c1r_{model}_iqp_scales.csv")
    Cb = pd.read_csv(TABLES / f"c1r_{model}_iqp_by_prebaseline.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.3), constrained_layout=True, gridspec_kw=dict(width_ratios=[1.25, 1.0, 1.05]))
    ax = axes[0]
    groups = ["ALL"] + DOMS
    adj = [("raw", "raw", style.TEAL, "o", True), ("pre", "pre-treatment baseline", style.TEAL, "s", False),
           ("anon", "post-treatment: anon world", style.TERRACOTTA, "^", True), ("none", "post-treatment: no-label world", style.TERRACOTTA, "D", False)]
    F = R[R.set == "full"]
    for gi, g in enumerate(groups):
        y0 = len(groups) - 1 - gi
        for k, (a, lab, c, m, fill) in enumerate(adj):
            r = F[(F.group == g) & (F.adjustment == a)].iloc[0]
            ci_h(ax, y0 + 0.27 - 0.18 * k, r.estimate, r.lower, r.upper, c, m, fill, label=lab if gi == 0 else None)
    ax.axvline(0, color=style.SLATE, lw=0.8)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([DLAB[g] for g in groups][::-1])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("$I_{QP}$ (logits)")
    ax.set_title("a  Logit scale, by baseline", loc="left")
    # b: probability scale by content correctness (pooled)
    ax = axes[1]
    cls = [("correct", "correct content", style.TEAL), ("wrong", "wrong content", style.TERRACOTTA), ("no_target", "no target", style.PLUM)]
    P = Sc[(Sc.set == "full") & (Sc.scale == "probability")]
    for gi, g in enumerate(groups):
        y0 = len(groups) - 1 - gi
        for k, (a, lab, c) in enumerate(cls):
            s = P[(P.group == g) & (P.advocacy == a)]
            if not len(s):
                continue
            r = s.iloc[0]
            ci_h(ax, y0 + 0.2 - 0.2 * k, 100 * r.estimate, 100 * r.lower, 100 * r.upper, c, label=lab if gi == 0 else None)
    ax.axvline(0, color=style.SLATE, lw=0.8)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("$I_{QP}$ on the probability scale (pp)")
    ax.set_title("b  Probability scale", loc="left")
    # c: interaction conditioned on the pre-treatment state
    ax = axes[2]
    c = Cb[(Cb.set == "full") & (Cb.group == "ALL") & (Cb.n >= 50)].sort_values("bin_mid")
    ax.fill_between(c.bin_mid, c.lower, c.upper, color=style.PLUM, alpha=0.18, lw=0)
    ax.plot(c.bin_mid, c.estimate, color=style.PLUM, marker="o", ms=4)
    ax.axhline(0, color=style.SLATE, lw=0.8)
    ax.set_xlabel("Pre-treatment margin toward the advocated\noption (neutral message, logits)")
    ax.set_ylabel("$I_{QP}$ (logits)")
    ax.set_title("c  By pre-treatment state", loc="left")
    h0, l0 = axes[0].get_legend_handles_labels()
    h1, l1 = axes[1].get_legend_handles_labels()
    fig.legend(h0 + h1, l0 + l1, loc="outside lower center", ncol=4, fontsize=8)
    style.finish(fig, axes, FIGS / f"V3_C1_iqp_{model}")


def fig_state(model):
    F = pd.read_csv(TABLES / f"c1r_{model}_state_within_item.csv")
    B = pd.read_csv(TABLES / f"c1r_{model}_state_bins.csv")
    Co = pd.read_csv(TABLES / f"c1r_{model}_state_composition.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.1), constrained_layout=True, gridspec_kw=dict(width_ratios=[1.15, 1.0, 0.95]))
    ax = axes[0]
    rows = [("base_none", "all"), ("base_none", "target"), ("base_none", "no_target"), ("base_pre", "all"), ("base_pre", "target"),
            ("base_pre", "no_target")]
    lab = {"base_none": "no-label", "base_pre": "pre-treat.", "all": "all", "target": "target", "no_target": "no target"}
    import re
    parse = lambda s: [float(x) for x in re.findall(r"-?\d+\.\d+(?:e-?\d+)?", s)] if isinstance(s, str) else list(s)
    for i, (b, sub) in enumerate(rows):
        y = len(rows) - 1 - i
        r = F[(F.baseline == b) & (F.subset == sub)].iloc[0]
        lo, hi = parse(r.slope_below_ci)
        ci_h(ax, y + 0.13, r.slope_below, lo, hi, style.TEAL, label="slope below −3" if i == 0 else None)
        lo, hi = parse(r.slope_above_ci)
        ci_h(ax, y - 0.13, r.slope_above, lo, hi, style.TERRACOTTA, "s", label="slope above −3" if i == 0 else None)
    ax.axvline(0, color=style.SLATE, lw=0.8)
    ax.axhline(2.5, color=style.GRID, lw=0.8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{lab[b]} · {lab[s]}" for b, s in rows][::-1])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Within-item slope of P on baseline")
    ax.set_title("a  Within-item hinge test", loc="left")
    ax = axes[1]
    for q, c in (("strong", style.TEAL), ("weak", style.MUTED)):
        s = B[(B.scale == "probability") & (B.qual == q)].sort_values("bin_mid")
        ax.fill_between(s.bin_mid, 100 * s.lower, 100 * s.upper, color=c, alpha=0.18, lw=0)
        ax.plot(s.bin_mid, 100 * s.estimate, color=c, marker="o", ms=4, label=f"{q} evidence")
    ax.set_xlabel("Independent baseline margin\n(no-label world, logits)")
    ax.set_ylabel("Expert − anonymous, p(adopt) (pp)")
    ax.set_title("b  Probability scale", loc="left")
    ax.set_ylim(-3, 64)
    mass = B.groupby("bin_mid").mass_p05.min().min()
    ax.text(0.03, 0.97, f"label mass ≥ {mass:.3f} in all bins", transform=ax.transAxes, ha="left", va="top",
            fontsize=7, color=style.SLATE)
    ax = axes[2]
    d = Co[Co.level == "domain"].pivot_table(index="bin_mid", columns="group", values="share").reindex(columns=DOMS)
    bottom = np.zeros(len(d))
    for dm in DOMS:
        ax.bar(d.index, d[dm], bottom=bottom, width=1.7, color=style.DOMAIN_COLOR[dm], edgecolor="white", lw=1.0,
               label=DLAB[dm])
        bottom += d[dm].to_numpy()
    ax.set_ylim(0, 1)
    ax.set_xlabel("Independent baseline margin (logits)")
    ax.set_ylabel("Share of rows")
    ax.set_title("c  Domain mix per bin", loc="left")
    ax.grid(axis="x", visible=False)
    hs, ls = [], []
    for a_ in axes:
        h, l = a_.get_legend_handles_labels()
        hs += h; ls += l
    fig.legend(hs, ls, loc="outside lower center", ncol=7, fontsize=7.5, handletextpad=0.4, columnspacing=1.0)
    style.finish(fig, axes, FIGS / f"V3_C1_state_{model}")


def fig_gain(model):
    G = pd.read_csv(TABLES / f"c1r_{model}_gain.csv")
    Rr = pd.read_csv(TABLES / f"c1r_{model}_gain_random.csv")
    Au = pd.read_csv(TABLES / f"c1r_{model}_gain_auroc.csv")
    G = G[(G.world_set == "disagreement") & (G.direction == "q")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), constrained_layout=True, gridspec_kw=dict(width_ratios=[1.2, 1.2, 0.8]))
    pos = [("answer", "decision token", style.TERRACOTTA, "o"), ("msg_end", "end of message", style.PLUM, "s"),
           ("arg_mid", "mid-argument", style.TEAL, "^")]
    for ax, qty, title in ((axes[0], "gain", "a  Credibility gain"), (axes[1], "halo", "b  Halo")):
        for lo_, hi_ in ((9, 14), (15, 21)):
            ax.axvspan(lo_, hi_, color=style.GOLD, alpha=0.16, lw=0)
        if qty == "gain":
            rr = Rr[Rr.position == "answer"].sort_values("layer")
            ax.fill_between(rr.layer, -rr.rand_abs_p95, rr.rand_abs_p95, color=style.MUTED, alpha=0.22, lw=0,
                            label="random directions (95%)")
        for p, lab, c, m in pos:
            s = G[(G.position == p) & (G.quantity == qty)].sort_values("layer")
            ax.fill_between(s.layer, s.lower, s.upper, color=c, alpha=0.15, lw=0)
            ax.plot(s.layer, s.estimate, color=c, marker=m, ms=3.8, label=lab)
        ax.axhline(0, color=style.SLATE, lw=0.8)
        ax.set_xlabel("Layer")
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("SD units of the quality read-out")
    axes[0].text(11.5, axes[0].get_ylim()[1], "hand-off", ha="center", va="top", fontsize=7, color=style.SLATE)
    axes[0].text(18, axes[0].get_ylim()[1], "merge", ha="center", va="top", fontsize=7, color=style.SLATE)
    ax = axes[2]
    for kind, lab, c, m in (("q", "quality (orth.)", style.TEAL, "o"), ("p", "provenance", style.TERRACOTTA, "s")):
        s = Au[(Au.position == "answer") & (Au.direction == kind)].sort_values("layer")
        ax.plot(s.layer, s.prov_auroc, color=c, marker=m, ms=3.8, label=lab)
    ax.axhline(0.5, color=style.SLATE, lw=0.8)
    ax.set_ylim(0.3, 1.02)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Expert-vs-anon AUROC")
    ax.set_title("c  Direction purity", loc="left")
    ax.legend(loc="center right", fontsize=7)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=4, fontsize=8)
    style.finish(fig, axes, FIGS / f"V3_C1_gain_{model}")


def fig_lockin(model):
    E = pd.read_csv(TABLES / f"c1r_{model}_lockin.csv").set_index(["group", "estimand"])
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), constrained_layout=True, gridspec_kw=dict(width_ratios=[1.2, 1.0, 1.0]))
    groups = ["ALL"] + DOMS
    ax = axes[0]
    for gi, g in enumerate(groups):
        y = len(groups) - 1 - gi
        t0 = E.loc[(g, "t0_majority_acc")]
        for k, (est, lab, c, m) in enumerate((("debate_t11_majority_acc", "debate, turn 11", style.TERRACOTTA, "o"),
                                              ("pool_t11_majority_acc", "independent pool, turn 11", style.TEAL, "s"))):
            r = E.loc[(g, est)]
            yy = y + 0.12 - 0.24 * k
            ci_h(ax, yy, 100 * r.estimate, 100 * r.lower, 100 * r.upper, c, m, label=lab if gi == 0 else None)
        ax.scatter([100 * t0.estimate], [y], s=30, facecolor="white", edgecolor=style.INK, lw=1.1, zorder=4,
                   label="turn-0 majority vote" if gi == 0 else None)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([DLAB[g] for g in groups][::-1])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Majority accuracy (%)")
    ax.set_title("a  Beyond majority vote", loc="left")
    ax = axes[1]
    conds = [("t0_maj_wrong", "ends_wrong_unanimous", "turn-0 majority wrong:\nends wrong & unanimous"),
             ("t0_maj_right", "ends_majority_correct", "turn-0 majority right:\nends correct")]
    for i, (cnd, out, lab) in enumerate(conds):
        y = 1 - i
        for k, (c_, col, m) in enumerate((("debate", style.TERRACOTTA, "o"), ("pool", style.TEAL, "s"))):
            r = E.loc[("ALL", f"{c_}_{out} | {cnd}")]
            ci_h(ax, y + 0.12 - 0.24 * k, 100 * r.estimate, 100 * r.lower, 100 * r.upper, col, m)
    ax.set_yticks([1, 0])
    ax.set_yticklabels([c[2] for c in conds], fontsize=7.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Items (%)")
    ax.set_title("b  Outcome given turn 0", loc="left")
    ax = axes[2]
    for gi, g in enumerate(groups):
        y = len(groups) - 1 - gi
        for k, (c_, col, m) in enumerate((("debate", style.TERRACOTTA, "o"), ("pool", style.TEAL, "s"))):
            key = (g, f"{c_}_t11_majority_equals_t0_majority | t0_split")
            if key in E.index:
                r = E.loc[key]
                ci_h(ax, y + 0.12 - 0.24 * k, 100 * r.estimate, 100 * r.lower, 100 * r.upper, col, m)
        if (g, "debate_t11_majority_equals_t0_majority | t0_split") in E.index:
            n = int(E.loc[(g, "debate_t11_majority_equals_t0_majority | t0_split")].n)
            ax.text(101, y, f"n={n}", va="center", fontsize=7, color=style.SLATE)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([])
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 112)
    ax.set_xlabel("Turn-11 majority = turn-0 majority (%)")
    ax.set_title("c  Turn-0 split items", loc="left")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=3, fontsize=8)
    style.finish(fig, axes, FIGS / f"V3_C1_lockin_{model}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Q9")
    ap.add_argument("--figs", default="iqp,state,gain,lockin")
    a = ap.parse_args()
    for f in a.figs.split(","):
        globals()[f"fig_{f}"](a.model)
        print("[fig]", f, flush=True)
