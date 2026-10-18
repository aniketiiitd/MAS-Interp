"""F4 evidence x provenance interaction diamonds, F5 causal footprint, F6 PCIF phase portrait (from frozen tables)."""
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


def f4_diamonds(model, run="cpt_core"):
    ic = pd.read_csv(TABLES / f"cpt_{model}_{run}_m_adv_itemcontrasts.csv")
    summ = pd.read_csv(TABLES / f"cpt_{model}_{run}_m_adv_summary.csv")
    order = [f for _, f, _ in FAMILY_ORDER]
    dom_of = {f: d for d, f, _ in FAMILY_ORDER}
    cells = ic.groupby("family")[["Y_dis_strong_exp", "Y_dis_strong_anon", "Y_dis_weak_exp", "Y_dis_weak_anon"]].mean()
    fig, (ax, axd, axn) = plt.subplots(1, 3, figsize=(7.2, 4.3), sharey=True, constrained_layout=True,
                                       gridspec_kw=dict(width_ratios=[2.3, 1.15, 0.28]))
    y = np.arange(len(order))[::-1]
    for yi, f in zip(y, order):
        if f not in cells.index:
            continue
        r = cells.loc[f]
        for q, c, dy in (("strong", style.TEAL, 0.17), ("weak", style.MUTED, -0.17)):
            a, e = r[f"Y_dis_{q}_anon"], r[f"Y_dis_{q}_exp"]
            ax.plot([a, e], [yi + dy, yi + dy], color=c, lw=1.5, zorder=2, solid_capstyle="round")
            ax.scatter([a], [yi + dy], s=20, facecolor="white", edgecolor=c, lw=1.1, zorder=3)
            ax.scatter([e], [yi + dy], s=22, facecolor=c, edgecolor=c, lw=1.1, zorder=3)
    ax.axvline(0, color=style.SLATE, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([FAMILY_LABEL[f] for f in order])
    ax.set_xlabel("Adoption margin when the message disagrees\n(log p advocated − log p other)")
    ax.grid(axis="y", visible=False)
    s = summ[(summ.level == "family") & (summ.contrast == "IQP")].set_index("group")
    axd.axvline(0, color=style.SLATE, lw=0.8)
    for yi, f in zip(y, order):
        if f not in s.index:
            continue
        rr = s.loc[f]
        c = style.PLUM
        sig = (rr.lower > 0) or (rr.upper < 0)
        axd.plot([rr.lower, rr.upper], [yi, yi], color=c, lw=1.3)
        axd.scatter([rr.estimate], [yi], marker="D", s=30, facecolor=c if sig else "white", edgecolor=c, lw=1.2, zorder=3)
    axd.set_xlabel("Interaction $I_{QP}$\n(expert gain: strong − weak)")
    axd.grid(axis="y", visible=False)
    n = s.n_items
    for yi, f in zip(y, order):
        axn.text(0, yi, f"{int(n.get(f, 0))}", va="center", fontsize=8.5, color=style.SLATE)
    axn.set_axis_off()
    for b in (4.5, 9.5):
        for a_ in (ax, axd):
            a_.axhline(len(order) - 1 - b, color=style.GRID, lw=0.8)
    handles = [plt.Line2D([], [], color=style.TEAL, lw=1.5, marker="o", mfc=style.TEAL, label="strong evidence"),
               plt.Line2D([], [], color=style.MUTED, lw=1.5, marker="o", mfc=style.MUTED, label="weak evidence"),
               plt.Line2D([], [], color=style.INK, lw=0, marker="o", mfc="white", label="anonymous source"),
               plt.Line2D([], [], color=style.INK, lw=0, marker="o", mfc=style.INK, label="expert source")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=8.5)
    style.finish(fig, [ax, axd], FIGS / f"F4_interaction_{model}_{run}")


def f4_headline(model, run="cpt_core"):
    """F4: credentials matter most when content is wrong. Left: P(switch) anonymous->expert for correct vs wrong advocacy
    (strong evidence), one row per target family plus the pooled row. Right: I_CP (correctness x provenance) with CIs."""
    A = pd.read_csv(TABLES / f"cpt_{model}_{run}_adoption.csv")
    summ = pd.read_csv(TABLES / f"cpt_{model}_{run}_m_adv_summary.csv")
    fams = [f for _, f, _ in FAMILY_ORDER if f in set(A[(A.level == "family") & (A.advocacy != "no_target")].group)]
    rows = [("ALL", "all", "All target items")] + [(f, "family", FAMILY_LABEL[f]) for f in fams]
    fig, (ax, axd) = plt.subplots(1, 2, figsize=(7.2, 4.0), sharey=True, constrained_layout=True,
                                  gridspec_kw=dict(width_ratios=[2.2, 1.0]))
    y = np.arange(len(rows))[::-1].astype(float)
    y[0] += 0.6  # visual gap under the pooled row
    for yi, (g, lvl, lab) in zip(y, rows):
        for adv, col, dy in (("correct", style.TEAL, 0.16), ("wrong", style.TERRACOTTA, -0.16)):
            s = A[(A.level == lvl) & (A.group == g) & (A.advocacy == adv) & (A.qual == "strong")].set_index("prov")
            if not {"anon", "expert"} <= set(s.index):
                continue
            a_, e_ = s.loc["anon", "p_adopt"], s.loc["expert", "p_adopt"]
            ax.annotate("", xy=(e_, yi + dy), xytext=(a_, yi + dy),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.5, shrinkA=3, shrinkB=2))
            ax.scatter([a_], [yi + dy], s=20, facecolor="white", edgecolor=col, lw=1.1, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, _, lab in rows])
    ax.set_xlim(0, 1)
    ax.set_xlabel("P(receiver switches to the advocated answer)\nopen = anonymous source, arrow head = expert source")
    ax.grid(axis="y", visible=False)
    s = summ[(summ.contrast == "ICP")]
    axd.axvline(0, color=style.SLATE, lw=0.8)
    for yi, (g, lvl, lab) in zip(y, rows):
        r = s[(s.level == lvl) & (s.group == g)]
        if not len(r):
            continue
        r = r.iloc[0]
        sig = (r.lower > 0) or (r.upper < 0)
        axd.plot([r.lower, r.upper], [yi, yi], color=style.PLUM, lw=1.3)
        axd.scatter([r.estimate], [yi], marker="D", s=30 if lvl == "family" else 46, facecolor=style.PLUM if sig else "white",
                    edgecolor=style.PLUM, lw=1.2, zorder=3)
    axd.set_xlabel("Correctness x provenance $I_{CP}$\n(logits; < 0: credentials help wrong answers more)")
    axd.grid(axis="y", visible=False)
    handles = [plt.Line2D([], [], color=style.TEAL, lw=1.5, label="message argues the correct answer"),
               plt.Line2D([], [], color=style.TERRACOTTA, lw=1.5, label="message argues a wrong answer")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, fontsize=8.5)
    style.finish(fig, [ax, axd], FIGS / f"F4_credentials_{model}_{run}")


def f4_state(model, run="cpt_core"):
    """F4 (revised): (A) credential effect vs receiver state, strong vs weak evidence; (B) state-adjusted evidence x provenance
    interaction by domain/family under the independent no-label baseline, with the anonymous-baseline estimate as a tick."""
    C = pd.read_csv(TABLES / f"cpt_{model}_{run}_state_curve.csv")
    R = pd.read_csv(TABLES / f"cpt_{model}_{run}_state_adjusted.csv")
    fig, (ax, axf) = plt.subplots(1, 2, figsize=(7.2, 3.6), constrained_layout=True, gridspec_kw=dict(width_ratios=[1.15, 1.0]))
    for q, col, lab in (("strong", style.TEAL, "strong evidence"), ("weak", style.MUTED, "weak evidence")):
        g = C[C.qual == q].sort_values("bin_mid")
        ax.fill_between(g.bin_mid, g.lower, g.upper, color=col, alpha=0.18, lw=0)
        ax.plot(g.bin_mid, g.P, color=col, lw=1.8, marker="o", ms=3.5)
        ax.text(g.bin_mid.iloc[-1] + 0.3, g.P.iloc[-1], lab, fontsize=8.5, va="center", color=style.INK)
    ax.axhline(0, color=style.SLATE, lw=0.7)
    ax.set_xlabel("Receiver state: margin toward the advocated answer\nwith no source label (logits; < 0 = resists)")
    ax.set_ylabel("Credential effect\n(expert − anonymous, logits)")
    ax.set_xlim(-9.5, 9.5)
    ax.set_ylim(bottom=0)
    rows = [("ALL", "all", "All items")] + [(d, "domain", DOMAIN_LABEL[d]) for d in DOMAIN_LABEL] + \
           [(f, "family", FAMILY_LABEL[f]) for _, f, _ in FAMILY_ORDER]
    Q = R[R.interaction == "I_QP"]
    y = np.arange(len(rows))[::-1].astype(float)
    y[:4] += 0.8
    for yi, (g, lvl, lab) in zip(y, rows):
        r_none = Q[(Q.level == lvl) & (Q.group == g) & (Q.adjustment == "none")]
        r_anon = Q[(Q.level == lvl) & (Q.group == g) & (Q.adjustment == "anon")]
        if not len(r_none):
            continue
        r = r_none.iloc[0]
        col = style.DOMAIN_COLOR.get(g, style.PLUM) if lvl == "domain" else style.PLUM
        robust = bool(r.robust_both_adjustments)
        axf.plot([r.lower, r.upper], [yi, yi], color=col, lw=1.3)
        axf.scatter([r.estimate], [yi], marker="D", s=26 if lvl == "family" else 40, facecolor=col if robust else "white", edgecolor=col, lw=1.1, zorder=3)
        if len(r_anon):
            axf.scatter([r_anon.iloc[0].estimate], [yi], marker="|", s=60, color=style.SLATE, zorder=2)
    axf.axvline(0, color=style.SLATE, lw=0.8)
    axf.set_yticks(y)
    axf.set_yticklabels([lab for _, _, lab in rows], fontsize=8)
    axf.set_xlabel("State-matched evidence x provenance\n(strong − weak credential effect, logits)")
    axf.grid(axis="y", visible=False)
    style.finish(fig, [ax, axf], FIGS / f"F4_state_{model}_{run}")


def f5_footprint(model, split="hero_48"):
    split = "testfake" if model == "Q08" else split
    rt = pd.read_csv(TABLES / f"patch_recovery_{model}_{split}.csv")
    groups = [("src", "source label"), ("arg", "evidence text"), ("post", "post-message"), ("answer", "decision token")]
    P = rt[(rt.kind == "prov") & (rt["mode"] == "patch") & (rt.direction == "both")]
    Rn = rt[(rt.kind == "prov") & (rt["mode"] == "random") & (rt.direction == "both")]
    L = int(P.layer.max()) + 1
    M = np.full((len(groups), L), np.nan)
    LO = np.full_like(M, np.nan)
    for gi, (g, _) in enumerate(groups):
        s = P[P.group == g].set_index("layer")
        for l in s.index:
            M[gi, l] = s.loc[l, "recovery"]
            LO[gi, l] = s.loc[l, "lower"]
    fig = plt.figure(figsize=(7.2, 3.3), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[0.9, 1.6], width_ratios=[1, 0.025])
    axm = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0], sharex=axm)
    cax = fig.add_subplot(gs[1, 1])
    norm = plt.matplotlib.colors.TwoSlopeNorm(vmin=-0.4, vcenter=0.0, vmax=1.1)
    im = ax.imshow(M, aspect="auto", cmap=style.DIVERGING, norm=norm, interpolation="nearest",
                   extent=(-0.5, L - 0.5, len(groups) - 0.5, -0.5))
    # a cell is outlined when its CI lower bound exceeds the magnitude of the norm-matched random-direction effect
    # (max |estimate|, |CI bound|) measured at the nearest random-control layer for the same token group
    for gi, (g, _) in enumerate(groups):
        rg = Rn[Rn.group == g]
        for l in range(L):
            if not np.isfinite(LO[gi, l]):
                continue
            thr = 0.0
            if len(rg):
                near = rg.iloc[(rg.layer - l).abs().argsort()[:1]].iloc[0]
                thr = max(abs(near.recovery), abs(near.lower), abs(near.upper))
            if LO[gi, l] > thr:
                ax.scatter([l], [gi], s=5, color="white" if M[gi, l] > 0.55 else style.INK, zorder=3, lw=0)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([lab for _, lab in groups])
    ax.set_xlabel("Layer")
    ax.grid(False)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Patch recovery of\nprovenance effect", fontsize=8.5)
    for g, c, lab in (("src", style.TERRACOTTA, "source label"), ("answer", style.TEAL, "decision token")):
        s = P[P.group == g].sort_values("layer")
        axm.fill_between(s.layer, s.lower, s.upper, color=c, alpha=0.18, lw=0)
        axm.plot(s.layer, s.recovery, color=c, lw=1.6, label=lab)
        axm.text(s.layer.iloc[-1] + 0.4, s.recovery.iloc[-1] + (0.12 if g == "answer" else 0.0), lab, color=style.INK, fontsize=8.5, va="center")
    qa = rt[(rt.kind == "qual") & (rt["mode"] == "patch") & (rt.direction == "both") & (rt.group == "answer")].sort_values("layer")
    if len(qa):
        axm.fill_between(qa.layer, qa.lower, qa.upper, color=style.PLUM, alpha=0.12, lw=0)
        axm.plot(qa.layer, qa.recovery, color=style.PLUM, lw=1.2, ls=(0, (2, 1.5)))
        axm.text(qa.layer.iloc[-1] + 0.4, 0.62, "evidence quality\n(decision token)", color=style.PLUM, fontsize=8, va="center")
    axm.axhline(0, color=style.SLATE, lw=0.6)
    axm.set_ylabel("Recovery")
    axm.set_ylim(-0.35, 1.25)
    plt.setp(axm.get_xticklabels(), visible=False)
    summ = json.load(open(TABLES / f"patch_summary_{model}_{split}.json"))
    if summ.get("src_handoff_layer") is not None and summ.get("answer_onset_layer") is not None:
        a, b = summ["src_handoff_layer"], summ["answer_onset_layer"]
        axm.axvspan(min(a, b) - 0.5, max(a, b) + 0.5, color=style.GOLD, alpha=0.25, lw=0)
    style.finish(fig, [ax, axm], FIGS / f"F5_footprint_{model}_{split}")


def f6_pcif(model, run="cpt_core"):
    ic = pd.read_csv(TABLES / f"cpt_{model}_{run}_m_adv_items.csv")
    ic = ic.dropna(subset=["Q", "P", "IQP", "S"])
    # normalize each coordinate by its within-item noise scale (MAD of template-level contrasts if available)
    lo_q, hi_q = ic.Q.quantile([0.01, 0.99])
    lo_p, hi_p = ic.P.quantile([0.01, 0.99])
    clipped = ((ic.Q < lo_q) | (ic.Q > hi_q) | (ic.P < lo_p) | (ic.P > hi_p)).mean()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), sharex=True, sharey=True, constrained_layout=True)
    thr = np.quantile(np.abs(ic.IQP), 0.9)
    for ax, (dom, c) in zip(axes, style.DOMAIN_COLOR.items()):
        g = ic[ic.domain == dom]
        size = 6 + 40 * (g.S - ic.S.min()) / (ic.S.max() - ic.S.min() + 1e-9)
        strong_int = np.abs(g.IQP) >= thr
        ax.scatter(g.Q.clip(lo_q, hi_q), g.P.clip(lo_p, hi_p), s=size, color=c, alpha=0.35, lw=0)
        ax.scatter(g.Q[strong_int].clip(lo_q, hi_q), g.P[strong_int].clip(lo_p, hi_p), s=size[strong_int], facecolor="none",
                   edgecolor=style.INK, lw=0.6)
        ax.axhline(0, color=style.SLATE, lw=0.6)
        ax.axvline(0, color=style.SLATE, lw=0.6)
        ax.set_title(DOMAIN_LABEL[dom], color=style.INK)
        ax.set_xlabel("Evidence sensitivity $Q$")
    axes[0].set_ylabel("Provenance sensitivity $P$")
    json.dump(dict(clipped_fraction=float(clipped), size="social susceptibility S", ring="top-decile |I_QP|"),
              open(FIGS / f"F6_pcif_{model}_{run}.caption.json", "w"))
    style.finish(fig, axes, FIGS / f"F6_pcif_{model}_{run}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="+")
    ap.add_argument("--model", default="Q9")
    a = ap.parse_args()
    for w in a.which:
        {"f4": f4_diamonds, "f4h": f4_headline, "f4s": f4_state, "f5": f5_footprint, "f6": f6_pcif}[w](a.model)
    print("ok")
