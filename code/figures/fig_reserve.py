"""Reserve contribution figures (release_600 discovery).

  V3_R1_regret_<tag>: post-capitulation regret asymmetry. (a) later behaviour of agents that gave in, split by whether the
      abandoned answer was right; (b) the unprinted pull toward the abandoned answer at the capitulation turn and one turn later,
      from three readers of the identical text; (c) AUROC for a later reversal by reader and transcript baselines.
  V3_R2_overcaution_<tag>: over-caution dissociation. (a) probe read-out distributions for over-cautious, genuine and helpful
      states; (b) position index R by reader and condition; (c) private re-query flips.
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

TAG = "Q9_release_600_T12"


def _dot(ax, y, est, lo, hi, color, filled=True, size=26, label=None):
    ax.hlines(y, lo, hi, color=color, lw=1.4, zorder=3)
    if filled:
        ax.scatter([est], [y], s=size, color=color, edgecolors=CREAM, linewidths=0.6, zorder=4, label=label)
    else:
        ax.scatter([est], [y], s=size, facecolors=CREAM, edgecolors=color, linewidths=1.1, zorder=4, label=label)


def fig_r1(tag=TAG):
    S = pd.read_csv(TABLES / f"r1_contrasts_{tag}.csv").set_index("quantity")
    A = pd.read_csv(TABLES / f"r1_auroc_reader_{tag}.csv")
    style.apply()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.1), gridspec_kw=dict(width_ratios=[1.0, 1.2, 1.0], wspace=1.15))
    off = 0.14
    # (a) behaviour
    ax = axes[0]
    items = [("reverts_t1", "back to it\nnext turn"), ("reverts_later", "back to it\nlater"), ("private_to_X", "back to it\nin private")]
    for k, (q, lab) in enumerate(items):
        r = S.loc[q]
        _dot(ax, k - off, 100 * r.right, 100 * r.right_lo, 100 * r.right_hi, TEAL, label="abandoned answer was right" if k == 0 else None)
        _dot(ax, k + off, 100 * r.wrong, 100 * r.wrong_lo, 100 * r.wrong_hi, TERRACOTTA, label="abandoned answer was wrong" if k == 0 else None)
    ax.set_yticks(range(len(items)), [l for _, l in items])
    ax.invert_yaxis()
    ax.set_xlabel("Agents that gave in (%)")
    ax.set_xlim(-2, 70)
    ax.set_title("a  What they do afterwards", loc="left")
    # (b) pull toward the abandoned answer by reader and time
    ax = axes[1]
    rows = [("participant", "participant"), ("obsmin", "observer"), ("para", "paraphrase")]
    ylab = []
    y = 0
    for t, tl in (("t", "turn of giving in"), ("t1", "one turn later")):
        for rd, rl in rows:
            r = S.loc[f"mX_{rd}_{t}"]
            _dot(ax, y - off, r.right, r.right_lo, r.right_hi, TEAL, filled=(rd == "participant"))
            _dot(ax, y + off, r.wrong, r.wrong_lo, r.wrong_hi, TERRACOTTA, filled=(rd == "participant"))
            ylab.append(f"{rl}")
            y += 1
        y += 0.6
    ypos = [0, 1, 2, 3.6, 4.6, 5.6]
    ax.set_yticks(ypos, ylab)
    ax.invert_yaxis()
    ax.axvline(0, color=INK, lw=0.6)
    for yy, tl in ((1, "turn of giving in"), (4.6, "one turn later")):
        ax.text(1.03, yy, tl, transform=ax.get_yaxis_transform(), rotation=270, va="center", ha="left", fontsize=7.2, color=SLATE)
    ax.set_xlabel("log p(abandoned) − log p(adopted)")
    ax.set_title("b  Unprinted pull back", loc="left")
    # (c) AUROC for reverting later
    ax = axes[2]
    order = [("pull_participant", "participant pull", TEAL, True), ("pull_observer", "observer pull", TEAL, False),
             ("pull_para", "paraphrase pull", TEAL, False), ("tfidf", "TF-IDF (message)", SLATE, True),
             ("public_hist", "public history", SLATE, True), ("neg_conf_t1", "stated confidence", SLATE, False)]
    A = A[A.scale == "family_adjusted"].set_index("predictor")
    for k, (p, lab, col, filled) in enumerate(order):
        if p in A.index:
            r = A.loc[p]
            _dot(ax, k, r.auroc, r.lower, r.upper, col, filled=filled)
    ax.set_yticks(range(len(order)), [o[1] for o in order])
    ax.invert_yaxis()
    ax.axvline(0.5, color=INK, lw=0.6, ls=(0, (3, 2)))
    ax.set_xlim(0.2, 1.02)
    ax.set_xlabel("AUROC for reverting (within family)")
    ax.set_title("c  Who can read it", loc="left")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.02), handletextpad=0.3, columnspacing=1.6)
    style.finish(fig, axes, str(FIGS / f"V3_R1_regret_{tag}"))


def fig_r2(tag=TAG):
    s = json.load(open(TABLES / f"r2_summary_{tag}.json"))
    St = pd.read_csv(TABLES / f"r2_states_{tag}.csv", usecols=["reader", "family", "vkey", "tkey", "caution_margin", "abstention_dom"])
    style.apply()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.5), gridspec_kw=dict(width_ratios=[1.15, 1.0, 1.0], wspace=0.75))
    # (a) unprinted caution margin distributions, refusal slice (participant, all conditions)
    ax = axes[0]
    P = St[St.reader == "participant"]
    groups = [("unsafe request, refused", (P.family == "unsafe_request") & (P.vkey == "refuse"), TERRACOTTA),
              ("safe request, refused", (P.family == "over_refusal") & (P.vkey == "refuse"), PLUM),
              ("safe request, assisted", (P.family == "over_refusal") & (P.vkey == "assist"), TEAL)]
    vals = [P[m].caution_margin.to_numpy(float) for _, m, _ in groups]
    allv = np.concatenate([v[np.isfinite(v)] for v in vals])
    lo, hi = np.percentile(allv, [0.5, 99.5])
    bins = np.linspace(lo, hi, 40)
    for k, ((lab, m, col), v) in enumerate(zip(groups, vals)):
        v = v[np.isfinite(v)]
        h, e = np.histogram(v, bins=bins, density=True)
        h = h / h.max() * 0.62
        c = (e[:-1] + e[1:]) / 2
        ax.fill_between(c, k, k + h, color=col, alpha=0.3, lw=0, step="mid")
        ax.step(c, k + h, where="mid", color=col, lw=1.0)
        ax.text(lo, k + 0.7, f"{lab} (n={len(v)})", fontsize=7.0, color=INK, va="bottom")
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_yticks([])
    ax.set_ylim(-0.08, 3.25)
    ax.set_xlabel("log p(refuse) − log p(assist)")
    ax.set_title("a  Unprinted refusal margin", loc="left")
    # (b) position index R
    ax = axes[1]
    T = pd.read_csv(TABLES / f"r2_position_{tag}.csv")
    specs = [("refusal", "caution_margin", "refusal, margin", PLUM), ("abstention", "caution_margin", "abstention, margin", TERRACOTTA),
             ("abstention", "abstention_dom", "abstention, probe", TERRACOTTA)]
    rowsdef = (("participant", "debate", "participant", True), ("obsmin", "debate", "observer", False),
               ("para", "debate", "paraphrase", False), ("participant", "pool", "pool", True))
    labels, ypos, y = [], [], 0
    for sl, meas, name, col in specs:
        for rd, cond, lab, filled in rowsdef:
            r = T[(T.slice == sl) & (T.measure == meas) & (T.reader == rd) & (T.cond == cond)]
            if len(r):
                r = r.iloc[0]
                _dot(ax, y, r.R, r.R_lo, r.R_hi, col if cond != "pool" else SLATE, filled=filled, size=20)
            labels.append(lab)
            ypos.append(y)
            y += 1
        ax.text(0.02, y - 4.55, name, transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=7.0, color=col)
        y += 0.9
    ax.set_yticks(ypos, labels, fontsize=7.4)
    ax.invert_yaxis()
    ax.axvline(0, color=TEAL, lw=0.8, ls=(0, (3, 2)))
    ax.axvline(1, color=TERRACOTTA, lw=0.8, ls=(0, (3, 2)))
    ax.set_xlim(-0.05, 1.12)
    ax.set_xlabel("position index R\n0 = helpful · 1 = warranted caution")
    ax.set_title("b  Over-caution is internal", loc="left")
    # (c) private re-query flips (final turn)
    ax = axes[2]
    b = s["behaviour"]
    rows = [("refusal", "a", "safe · refused"), ("refusal", "b", "unsafe · refused"),
            ("abstention", "a", "answerable ·\nabstained"), ("abstention", "b", "unanswerable ·\nabstained")]
    for k, (sl, g, lab) in enumerate(rows):
        for j, (cd, col) in enumerate((("debate", TEAL), ("pool", SLATE))):
            key = f"{sl}|{cd}|{g}|private_flip"
            if key in b:
                e, l_, h_, n, nc = b[key]
                _dot(ax, k + (j - 0.5) * 0.3, 100 * e, 100 * l_, 100 * h_, col, size=20, label=cd if k == 0 else None)
                ax.text(101, k + (j - 0.5) * 0.3, f"n={n}", fontsize=6.4, color=SLATE, va="center")
    ax.set_yticks(range(len(rows)), [r[2] for r in rows], fontsize=7.4)
    ax.invert_yaxis()
    ax.set_xlim(-2, 100)
    ax.set_xlabel("private answer turns helpful (%)")
    ax.set_title("c  Asked privately", loc="left")
    ax.legend(loc="lower right", frameon=False, fontsize=7.2)
    style.finish(fig, axes, str(FIGS / f"V3_R2_overcaution_{tag}"))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("r1", "both"):
        fig_r1()
    if which in ("r2", "both"):
        fig_r2()
