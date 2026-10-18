"""What moves an agent in debate is destabilisation, not moral stance.

Two concept probes built by the identical pipeline on the same 100 topics -- same stories format, same mean pooling,
same within-topic centring, same topic-held-out folds, same permutation null (concept labels shuffled within topic and
the whole fit repeated). One reads 171 emotion concepts, the other 22 moral concepts.

  a  both instruments work, and the moral one works better: out-of-fold AUROC on the story task against its own
     permuted-label null
  b  only the emotion probe predicts behaviour, only under debate, and only at shallow-to-middle depth: share of
     concepts whose reading-site projection beats the permuted null at predicting that this agent is about to change
     its answer
  c  at layer 12 the effect is one signed dimension -- destabilised states predict switching, settled states predict
     holding -- and the moral concepts sit at the null

Usage: python figures/fig_concept.py [MODEL]
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "figures"))
import style  # noqa: E402
from common import TABLES  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "outputs/figures/paper"
S_, M_ = style.SLATE, style.MUTED
EMO, MOR = style.TERRACOTTA, style.TEAL
SHOW = ["shaken", "horrified", "terrified", "alarmed", "panicked", "unnerved", "humiliated", "ashamed", "anxious",
        "grateful", "joyful", "optimistic", "proud", "pleased", "cheerful", "happy", "energized", "satisfied"]


def main(m="Q9"):
    style.apply_paper()
    fig, ax = style.strip(3, 2.0, [0.86, 1.02, 1.12])

    # a -- instrument quality on the story task
    for k, (tag, col, lab) in enumerate((("emotion_stories", EMO, "emotion (171)"), ("moral_stories", MOR, "moral (22)"))):
        T = pd.read_csv(TABLES / f"concept_null_{tag}_{m}.csv")
        b = T.loc[T.groupby("concept").auroc.idxmax()]
        ax[0].scatter(np.full(len(b), k) + np.random.default_rng(3).normal(0, 0.055, len(b)), b.auroc,
                      s=5, color=col, alpha=0.45, lw=0, zorder=3)
        ax[0].plot([k - 0.24, k + 0.24], [b.auroc.median()] * 2, color=col, lw=1.6, zorder=4)
        ax[0].scatter(np.full(len(b), k) + 0.30, T.groupby("concept").null_median.median().reindex(b.concept),
                      s=4, color=M_, alpha=0.45, lw=0, zorder=3)
    ax[0].axhline(0.5, color=S_, lw=0.6, ls=(0, (3, 2)))
    ax[0].set_xticks([0, 1])
    ax[0].set_xticklabels(["emotion\n171", "moral\n22"])
    ax[0].set_xlim(-0.5, 1.6)
    ax[0].set_ylim(0.35, 1.02)
    ax[0].set_ylabel("out-of-fold AUROC")
    ax[0].set_title("a  Both probes work", loc="left")
    ax[0].text(0.97, 0.06, "grey = permuted-label null", transform=ax[0].transAxes, ha="right", va="bottom",
               fontsize=5.4, color=M_)

    # b -- share of concepts beating the null at predicting the next answer change
    # Plot significance, not the raw share: with only 22 moral concepts each hit is 4.5%, so a share axis makes
    # 2 of 22 (binomial p = 0.26) look level with 32 of 171 (p = 1.2e-10). -log10 p against the 5% null does not.
    from scipy.stats import binomtest
    for tag, col, mk, lab in (("emotion_stories", EMO, "o", "emotion"), ("moral_stories", MOR, "s", "moral")):
        T = pd.read_csv(TABLES / f"lit_concept_behaviour_{tag}_{m}.csv")
        for cond, ls in (("debate", "-"), ("reflect", ":")):
            g = T[T.cond == cond].groupby("layer").beats_null_p95.agg(["sum", "count"])
            y = [-np.log10(binomtest(int(r["sum"]), int(r["count"]), 0.05, alternative="greater").pvalue)
                 for _, r in g.iterrows()]
            ax[1].plot(g.index, y, color=col, ls=ls, marker=mk if cond == "debate" else None,
                       ms=3.0, lw=1.2 if cond == "debate" else 0.9,
                       label=f"{lab}, {'debate' if cond == 'debate' else 'no comm.'}")
    ax[1].axhline(-np.log10(0.05), color=S_, lw=0.6, ls=(0, (3, 2)))
    ax[1].text(31, -np.log10(0.05) + 0.3, "p = .05", fontsize=5.4, color=S_, ha="right")
    ax[1].set_xlabel("layer")
    ax[1].set_ylabel("enrichment over null, $-\\log_{10}p$")
    ax[1].set_title("b  Only emotion predicts the switch", loc="left")
    ax[1].legend(loc="upper right", fontsize=5.4, frameon=False, labelspacing=0.2, handletextpad=0.4)

    # c -- the signed dimension at layer 12
    E = pd.read_csv(TABLES / f"lit_concept_behaviour_emotion_stories_{m}.csv")
    E = E[(E.layer == 12) & (E.cond == "debate")].set_index("concept")
    present = [c for c in SHOW if c in E.index]
    vals = [E.loc[c].auroc for c in present]
    order = np.argsort(vals)
    for y, i in enumerate(order):
        c, v = present[i], vals[i]
        ax[2].plot([0.5, v], [y, y], color=EMO if v > 0.5 else M_, lw=1.0, solid_capstyle="round", zorder=2)
        ax[2].scatter([v], [y], s=13, color=EMO if v > 0.5 else M_, zorder=3)
    Mo = pd.read_csv(TABLES / f"lit_concept_behaviour_moral_stories_{m}.csv")
    Mo = Mo[(Mo.layer == 12) & (Mo.cond == "debate")]
    ax[2].scatter(Mo.auroc, np.full(len(Mo), len(present) + 0.9), s=9, color=MOR, alpha=0.8, zorder=3)
    ax[2].text(0.5, len(present) + 1.9, "22 moral concepts", fontsize=5.4, color=MOR, ha="center")
    ax[2].axvline(0.5, color=S_, lw=0.7)
    ax[2].set_yticks(range(len(present)))
    ax[2].set_yticklabels([present[i] for i in order], fontsize=5.2)
    ax[2].set_ylim(-0.8, len(present) + 2.6)
    ax[2].grid(axis="y", visible=False)
    ax[2].set_xlabel("AUROC, predicts the coming switch")
    ax[2].set_title("c  One signed dimension (layer 12)", loc="left")
    ax[2].text(0.02, 0.20, "settled\n→ holds firm", transform=ax[2].transAxes, fontsize=5.4, color=M_, va="center")
    ax[2].text(0.98, 0.10, "destabilised\n→ switches", transform=ax[2].transAxes, fontsize=5.4, color=EMO,
               va="center", ha="right")
    style.finish_paper(fig, ax, f"{OUT}/fig_concept_{m}")
    print(f"wrote {OUT}/fig_concept_{m}.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Q9")
