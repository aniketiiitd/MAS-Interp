"""V3-C3 figures: role fingerprint (participant - observer DiD over validated concepts) and behavioural role effect B.

  V3_C3_fingerprint_<tag>: two wide strips (decision site, reading site) of the full-sample DiD for every validated concept,
      grouped by taxonomy, filled = replicates discovery -> confirmation, shaded band = paraphrase-null 95th percentile, dashed =
      random-direction 95th percentile; bottom: the 8 pre-registered concepts on the confirmation half, observer contrast vs the
      paraphrase null.
  V3_C3_behaviour_<tag>: outnumbered episodes, P(argmax = peers' shared answer) and margin toward it, by reader and domain.
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
from style import CREAM, GRID, INK, MUTED, PLUM, SLATE, TEAL, TERRACOTTA, plt

PREREG = ["dissent", "conflicted", "moral_uncertainty", "deferential", "trust_in_source", "belief_updating",
          "conformity_to_majority", "stubbornness"]
GROUP_ORDER = ["social_epistemic", "moral", "epistemic", "social_tension", "self_conscious", "fear_anxiety", "anger", "aversion",
               "sadness", "low_energy", "social_warmth", "agency_positive", "positive_low_arousal", "positive_high_arousal"]
GROUP_SHORT = {"social_epistemic": "social-\nepistemic", "moral": "moral", "epistemic": "epistemic", "social_tension": "social\ntension",
               "self_conscious": "self-\nconscious", "fear_anxiety": "fear", "anger": "anger", "aversion": "aversion",
               "sadness": "sadness", "low_energy": "low\nenergy", "social_warmth": "warmth", "agency_positive": "agency",
               "positive_low_arousal": "calm", "positive_high_arousal": "joy"}
SITE_LABEL = {"dec": "decision token", "read": "end of peer messages"}
DOM_SHORT = {"knowledge_reasoning": "Knowledge", "values_moral_judgment": "Values", "safety_refusal": "Safety"}


def pretty(c):
    return c.replace("_", " ")


def fingerprint(tag, variant="general"):
    F = pd.read_csv(TABLES / f"inner_fingerprint_{tag}.csv")
    F = F[F.variant == variant]
    style.apply()
    fig = plt.figure(figsize=(7.2, 7.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.25], hspace=0.66, wspace=0.55)
    for r, site in enumerate(("dec", "read")):
        ax = fig.add_subplot(gs[r, :])
        T = F[(F.site == site) & F.validated].copy()
        T["g"] = pd.Categorical(T.group, GROUP_ORDER, ordered=True)
        T = T.sort_values(["g", "full_est"]).reset_index(drop=True)
        x = np.arange(len(T))
        # group bands and labels (below the strip, alternating rows so narrow groups never collide)
        spans = T.groupby("g", observed=True).apply(lambda d: (d.index.min(), d.index.max())).to_dict()
        for i, (g, (a, b)) in enumerate(spans.items()):
            if i % 2 == 0:
                ax.axvspan(a - 0.5, b + 0.5, color="white", alpha=0.55, lw=0, zorder=0)
            ax.text((a + b) / 2, -0.05 - 0.11 * (i % 2), GROUP_SHORT.get(g, g).replace("-\n", "-").replace("\n", " "), transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=6.6, color=SLATE)
        if "para_null_p95_full" in T and T.para_null_p95_full.notna().any():
            p95 = T.para_null_p95_full.iloc[0]
            ax.axhspan(-p95, p95, color=MUTED, alpha=0.22, lw=0, zorder=1)
        if "random_full_abs_p95" in T and T.random_full_abs_p95.notna().any():
            r95 = T.random_full_abs_p95.iloc[0]
            for s_ in (-1, 1):
                ax.axhline(s_ * r95, color=SLATE, lw=0.7, ls=(0, (3, 2)), zorder=1)
        ax.axhline(0, color=INK, lw=0.6, zorder=2)
        rep = T.replicates_uncapped.fillna(False).to_numpy(bool)
        pre = T.concept.isin(PREREG).to_numpy()
        ax.vlines(x, T.full_lo, T.full_hi, color=MUTED, lw=0.6, zorder=3)
        col = np.where(pre, PLUM, TEAL)
        ax.scatter(x[rep], T.full_est[rep], s=11, c=col[rep], edgecolors=CREAM, linewidths=0.5, zorder=4)
        ax.scatter(x[~rep], T.full_est[~rep], s=11, facecolors=CREAM, edgecolors=col[~rep], linewidths=0.8, zorder=4)
        lo_, hi_ = np.nanmin(T.full_lo), np.nanmax(T.full_hi)
        rng_ = hi_ - lo_
        ax.set_ylim(lo_ - 0.55 * rng_, hi_ + 0.55 * rng_)
        top = T.assign(a=T.full_est.abs()).sort_values("a", ascending=False).head(5).index
        placed = []
        for i in sorted(top):
            y = T.full_est[i]
            up = bool(y >= 0)
            k = min(2, sum(1 for (xi, ui) in placed if abs(xi - i) < 26 and ui == up))
            placed.append((i, up))
            ty = (hi_ + (0.08 + 0.16 * k) * rng_) if up else (lo_ - (0.08 + 0.16 * k) * rng_)
            ax.annotate(pretty(T.concept[i]), xy=(x[i], T.full_hi[i] if up else T.full_lo[i]), xytext=(x[i] + 2.5, ty),
                        textcoords="data", ha="left", va="center", fontsize=6.4, color=INK,
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=0, shrinkB=1))
        ax.set_xticks([])
        ax.set_xlim(-1, len(T))
        ax.set_ylabel("DiD (z)\noutnumbered − majority")
        ax.set_title(f"{'ab'[r]}   Participant − observer, {SITE_LABEL[site]}: all {len(T)} validated concepts", loc="left")
        ax.grid(axis="x", visible=False)
        if r == 1:
            from matplotlib.lines import Line2D
            from matplotlib.patches import Patch
            hs = [Line2D([], [], ls="", marker="o", ms=4, color=TEAL, mec=CREAM, label="replicates (discovery → confirmation)"),
                  Line2D([], [], ls="", marker="o", ms=4, mfc=CREAM, mec=TEAL, label="does not replicate"),
                  Line2D([], [], ls="", marker="o", ms=4, color=PLUM, mec=CREAM, label="pre-registered"),
                  Patch(color=MUTED, alpha=0.3, label="paraphrase null, 95th pct")]
            if "random_full_abs_p95" in T and T.random_full_abs_p95.notna().any():
                hs.append(Line2D([], [], color=SLATE, lw=0.7, ls=(0, (3, 2)), label="random directions, 95th pct"))
            ax.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=len(hs), fontsize=6.8, handletextpad=0.3,
                      columnspacing=0.9)
    # prereg forests
    for c_, site in enumerate(("dec", "read")):
        ax = fig.add_subplot(gs[2, c_])
        T = F[(F.site == site)].set_index("concept").loc[PREREG[::-1]]
        y = np.arange(len(T))
        if "para_conf_est" in T:
            ax.hlines(y - 0.16, T.para_conf_lo, T.para_conf_hi, color=TERRACOTTA, lw=1.0)
            ax.scatter(T.para_conf_est, y - 0.16, s=16, color=TERRACOTTA, edgecolors=CREAM, linewidths=0.5, zorder=3,
                       label="paraphrased participant (wording null)")
        ok = (T.conf_same_sign_excl0 & T.get("exceeds_para_null", False)).to_numpy(bool)
        ax.hlines(y + 0.16, T.conf_lo, T.conf_hi, color=TEAL, lw=1.0)
        ax.scatter(T.conf_est[ok], (y + 0.16)[ok], s=18, color=TEAL, edgecolors=CREAM, linewidths=0.5, zorder=3,
                   label="observer (passes gate)")
        ax.scatter(T.conf_est[~ok], (y + 0.16)[~ok], s=18, facecolors=CREAM, edgecolors=TEAL, linewidths=0.9, zorder=3,
                   label="observer (fails gate)")
        if "para_null_p95_conf" in T:
            p95 = T.para_null_p95_conf.iloc[0]
            ax.axvspan(-p95, p95, color=MUTED, alpha=0.22, lw=0)
        ax.axvline(0, color=INK, lw=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels([pretty(c) + ("" if T.validated[c] else " †") for c in T.index], fontsize=8)
        ax.set_xlabel("DiD on the confirmation half (z)")
        ax.set_title(f"{'cd'[c_]}   Pre-registered concepts, {SITE_LABEL[site]}", loc="left")
        ax.grid(axis="y", visible=False)
        if c_ == 0:
            ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.28), ncol=3, fontsize=7.6, handletextpad=0.3, columnspacing=1.2)
            ax.text(0.0, -0.44, "† pre-registered but not story-validated.  Grey band: 95th percentile of |DiD| of the paraphrase null.",
                    transform=ax.transAxes, fontsize=7.0, color=SLATE, va="top")
    stem = FIGS / f"V3_C3_fingerprint_{tag}"
    style.finish(fig, fig.axes, str(stem))
    return stem


def behaviour(tag):
    B = pd.read_csv(TABLES / f"inner_behaviour_{tag}.csv")
    S = json.load(open(TABLES / f"inner_summary_{tag}.json"))
    B = B[B.role == "outnumbered"]
    readers = [("participant", "participant", INK, True), ("obsmin", "observer", TEAL, True), ("para", "paraphrased participant", TERRACOTTA, True),
               ("stakes", "observer, v1 wording", PLUM, False)]
    readers = [r for r in readers if (B.quantity == f"adopt_{r[0]}").any()]
    scopes = [s for s in ["ALL"] + list(DOM_SHORT) if (B.scope == s).any()]
    style.apply()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), sharey=True, gridspec_kw=dict(wspace=0.12))
    for ax, scale, xl in ((axes[0], "adopt", "P(verdict = peers' shared answer)"), (axes[1], "margin", "log p(peers' answer) − log p(own answer)")):
        for si, sc in enumerate(scopes):
            g = B[B.scope == sc].set_index("quantity")
            for ri, (key, lab, col, filled) in enumerate(readers):
                q = f"{scale}_{key}"
                if q not in g.index:
                    continue
                yy = -si + 0.27 - ri * 0.18
                r = g.loc[q]
                ax.hlines(yy, r.lower, r.upper, color=col, lw=1.1)
                ax.scatter([r.estimate], [yy], s=22, color=col if filled else CREAM, edgecolors=col if not filled else CREAM,
                           linewidths=0.9 if not filled else 0.5, zorder=3, label=lab if si == 0 else None)
            if scale == "adopt" and "adopt_sampled_participant" in g.index:
                r = g.loc["adopt_sampled_participant"]
                ax.scatter([r.estimate], [-si + 0.27 + 0.16], marker="|", s=40, color=SLATE, zorder=3,
                           label="participant, sampled text" if si == 0 else None)
        ax.set_yticks([-i for i in range(len(scopes))])
        ax.set_yticklabels([("All items" if s == "ALL" else DOM_SHORT[s]) + f"\n(n={int(B[(B.scope == s)].n.iloc[0])})" for s in scopes])
        ax.set_xlabel(xl)
        ax.grid(axis="y", visible=False)
        if scale == "margin":
            ax.axvline(0, color=INK, lw=0.6)
    g1 = S["gate_i_this_dataset"]
    txt = f"participant − observer B = {100*g1['B_adopt'][0]:+.1f} pp [{100*g1['B_adopt'][1]:+.1f}, {100*g1['B_adopt'][2]:+.1f}]"
    if g1.get("B_para_adopt"):
        txt += f"\nparticipant − paraphrase = {100*g1['B_para_adopt'][0]:+.1f} pp [{100*g1['B_para_adopt'][1]:+.1f}, {100*g1['B_para_adopt'][2]:+.1f}]"
    if g1.get("absB_minus_absBpara_adopt"):
        d = g1["absB_minus_absBpara_adopt"]
        txt += f"\n|B| − |wording null| = {100*d[0]:+.1f} pp [{100*d[1]:+.1f}, {100*d[2]:+.1f}]"
    axes[1].text(1.0, 1.13, txt.replace("\n", "    "), transform=axes[1].transAxes, fontsize=7.2, color=SLATE, va="bottom", ha="right")
    axes[0].set_title("a   Adopting the united peers' answer", loc="left")
    axes[1].set_title("b   Margin toward it", loc="left")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0.0, -0.2), ncol=5, fontsize=7.4, handletextpad=0.3, columnspacing=1.0)
    stem = FIGS / f"V3_C3_behaviour_{tag}"
    style.finish(fig, axes, str(stem))
    return stem


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "Q9_release_600_T12"
    print(fingerprint(tag), behaviour(tag))
