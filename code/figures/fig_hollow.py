"""What the agents wrote vs what their states held, at two read-out sites on byte-identical text.

Panel a: predicting which agent in a unanimous public group will privately give a different answer.
Panel b: predicting whether that unanimous answer is wrong.
Panel c: the contrast that carries the result -- the answer token the agent itself wrote against a fresh forced-answer
         read of the same finished transcript, under three prompt framings (participant, reworded participant, outside
         reader). The three framings agree, so the gap is about the read-out site, not about role or wording.

Usage: python figures/fig_hollow.py [MODEL]
"""
from __future__ import annotations

import sys

import pandas as pd

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "figures"))
import style  # noqa: E402
from common import TABLES  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "outputs/figures/paper"
S_, M_ = style.SLATE, style.MUTED

ROWS = [
    ("neg_margin", "the answer token it wrote", style.TERRACOTTA, "o"),
    ("neg_margin_part", "fresh read — participant", style.TEAL, "s"),
    ("neg_margin_part_para", "fresh read — participant, reworded", style.TEAL, "D"),
    ("neg_margin_observer", "fresh read — outside reader", style.TEAL, "^"),
    ("neg_hedge", "hedging words in its text", M_, "v"),
    ("public_hist", "behaviour features of the transcript", S_, "P"),
    ("tfidf", "text model of the transcript", S_, "X"),
]
CONTRASTS = [
    ("neg_margin - neg_margin_part", "wrote  −  fresh, participant"),
    ("neg_margin - neg_margin_part_para", "wrote  −  fresh, reworded"),
    ("neg_margin - neg_margin_observer", "wrote  −  fresh, outside reader"),
    ("neg_margin_part - neg_margin_observer", "participant − outside reader"),
    ("neg_margin_part_para - neg_margin_observer", "reworded − outside reader"),
]


def hdot(ax, y, e, lo, hi, c, m, label=None, filled=True):
    ax.plot([lo, hi], [y, y], color=c, lw=1.0, solid_capstyle="round", zorder=2)
    ax.scatter([e], [y], s=15, marker=m, facecolor=c if filled else "white", edgecolor=c, lw=0.9, zorder=3, label=label)


def auroc_panel(ax, T, title, xlabel):
    present = [r for r in ROWS if (T.predictor == r[0]).any()]
    for k, (key, lab, c, mk) in enumerate(present):
        r = T[T.predictor == key].iloc[0]
        hdot(ax, len(present) - 1 - k, r.auroc, r.lower, r.upper, c, mk)
    ax.axvline(0.5, color=S_, lw=0.6, ls=(0, (3, 2)))
    ax.set_yticks(range(len(present)))
    ax.set_yticklabels([lab for _, lab, _, _ in present][::-1])
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.6, len(present) - 0.4)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")


def main(m="Q9", dataset="lit1200"):
    style.apply_paper()
    A1 = pd.read_csv(TABLES / f"lit_hollow_{m}_{dataset}_auroc_defect.csv")
    A2 = pd.read_csv(TABLES / f"lit_hollow_{m}_{dataset}_auroc_wrong.csv")
    D1 = pd.read_csv(TABLES / f"lit_hollow_{m}_{dataset}_auroc_defect_diffs.csv")
    A1, A2, D1 = [d[d.scale == "family_adjusted"] for d in (A1, A2, D1)]
    fig, ax = style.strip(3, 1.95, [1.08, 0.60, 0.86])
    auroc_panel(ax[0], A1, "a  Private disagreement", "AUROC (family-adjusted)")
    auroc_panel(ax[1], A2, "b  Wrong consensus", "AUROC")
    ax[1].set_yticklabels([])
    present = [c for c in CONTRASTS if (D1.contrast == c[0]).any()]
    for k, (key, lab) in enumerate(present):
        r = D1[D1.contrast == key].iloc[0]
        sig = r.lower > 0 or r.upper < 0
        hdot(ax[2], len(present) - 1 - k, r["diff"], r.lower, r.upper,
             style.TERRACOTTA if sig else M_, "o", filled=sig)
    ax[2].axvline(0, color=S_, lw=0.6)
    ax[2].set_yticks(range(len(present)))
    ax[2].set_yticklabels([lab for _, lab in present][::-1])
    ax[2].grid(axis="y", visible=False)
    ax[2].set_ylim(-0.6, len(present) - 0.4)
    ax[2].set_xlabel("difference in AUROC")
    ax[2].set_title("c  Site, not role", loc="left")
    style.finish_paper(fig, ax, f"{OUT}/fig_hollow_{m}")
    print("wrote", f"{OUT}/fig_hollow_{m}.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Q9")
