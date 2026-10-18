"""Self-anchoring in debate is about the ownership label, not about being a participant.

Round-1 episodes where both peers agree on an answer the agent did not give (n=185, 177 item clusters). Five prompts
over byte-identical peer text. Two of them name the agent's own previous answer as its own; three present all three
responses as an unlabelled pool. `neutral_label` is the arm that separates the two accounts: it is a PARTICIPANT
instruction ("revise your recent opinion") carrying no ownership label, so the role account puts it with the
participants and the ownership account puts it with the outside readers.

Usage: python figures/fig_selfanchor.py [MODEL]
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

ARMS = [
    ("part", "participant, own answer named", True, "o"),
    ("part_para", "same, reworded", True, "D"),
    ("neutral_label", "participant, unlabelled", False, "s"),
    ("observer", "outside reader", False, "^"),
    ("observer_persona", "outside reader, reviewer", False, "v"),
]


def boot_contrasts(D, B=4000, seed=7):
    by = {c: g for c, g in D.groupby("cluster")}
    keys = list(by)
    rng = np.random.default_rng(seed)

    def st(d):
        P = (d.adopt_part + d.adopt_part_para).mean() / 2
        O = (d.adopt_observer + d.adopt_observer_persona).mean() / 2
        N = d.adopt_neutral_label.mean()
        return np.array([N - P, N - O])
    draws = np.array([st(pd.concat([by[keys[i]] for i in rng.integers(0, len(keys), len(keys))])) for _ in range(B)])
    return st(D), np.percentile(draws, 2.5, axis=0), np.percentile(draws, 97.5, axis=0)


def hdot(ax, y, e, lo, hi, c, m, filled=True, s=16):
    ax.plot([lo, hi], [y, y], color=c, lw=1.0, solid_capstyle="round", zorder=2)
    ax.scatter([e], [y], s=s, marker=m, facecolor=c if filled else "white", edgecolor=c, lw=0.9, zorder=3)


def main(m="Q9", dataset="lit1200"):
    style.apply_paper()
    A = pd.read_csv(TABLES / f"lit_role2x2_{m}_{dataset}_arms.csv")
    A = A[(A.scale == "adopt") & (A.domain == "all")].set_index("arm")
    D = pd.read_csv(TABLES / f"lit_role2x2_{m}_{dataset}_episodes.csv")
    O = pd.read_csv(TABLES / f"lit_frames_{m}_{dataset}_own.csv")
    fig, ax = style.strip(3, 1.85, [1.34, 0.66, 0.80])

    # a: the five arms, grouped by whether the agent's own previous answer is labelled as its own
    for k, (arm, lab, labelled, mk) in enumerate(ARMS):
        r = A.loc[arm]
        c = style.TERRACOTTA if labelled else style.TEAL
        hdot(ax[0], len(ARMS) - 1 - k, 100 * r.est, 100 * r.lower, 100 * r.upper, c, mk)
    ax[0].axhline(len(ARMS) - 2.5, color=style.GRID, lw=0.8)
    ax[0].set_yticks(range(len(ARMS)))
    ax[0].set_yticklabels([l for _, l, _, _ in ARMS][::-1])
    for lab, (_, _, labelled, _) in zip(ax[0].get_yticklabels(), ARMS[::-1]):
        lab.set_color(style.TERRACOTTA if labelled else style.TEAL)
    ax[0].grid(axis="y", visible=False)
    ax[0].set_ylim(-0.6, len(ARMS) - 0.4)
    ax[0].set_xlabel("adopts the peers' answer (%)")
    ax[0].set_title("a  Five prompts, identical peer text", loc="left")

    # b: where the discriminating arm lands
    est, lo, hi = boot_contrasts(D)
    labs = ["vs named-own\nparticipants", "vs outside\nreaders"]
    for k in range(2):
        sig = lo[k] > 0 or hi[k] < 0
        hdot(ax[1], 1 - k, 100 * est[k], 100 * lo[k], 100 * hi[k], style.TERRACOTTA if sig else M_, "o", filled=sig, s=20)
    ax[1].axvline(0, color=S_, lw=0.6)
    ax[1].set_yticks([0, 1])
    ax[1].set_yticklabels(labs[::-1])
    ax[1].grid(axis="y", visible=False)
    ax[1].set_ylim(-0.6, 1.4)
    ax[1].set_xlabel("\u0394 adoption (pts)")
    ax[1].set_title("b  The unlabelled one", loc="left")

    # c: the same ownership effect measured directly by swapping the label on identical text
    groups = [("all", "all regimes"), ("knowledge_reasoning", "Knowledge"), ("safety_refusal", "Safety"),
              ("values_moral_judgment", "Values")]
    for gi, (d, _) in enumerate(groups):
        for k, (metric, c, mk) in enumerate((("delta_own", style.TERRACOTTA, "o"), ("position_last_minus_first", M_, "s"))):
            r = O[(O.domain == d) & (O.metric == metric)]
            if not len(r):
                continue
            r = r.iloc[0]
            hdot(ax[2], len(groups) - 1 - gi + (0.15 if k == 0 else -0.15), r.est, r.lower, r.upper, c, mk, s=13)
    ax[2].axvline(0, color=S_, lw=0.6)
    ax[2].set_yticks(range(len(groups)))
    ax[2].set_yticklabels([l for _, l in groups][::-1])
    ax[2].grid(axis="y", visible=False)
    ax[2].set_ylim(-1.35, len(groups) - 0.4)
    ax[2].set_xlabel("\u0394 pull when labelled own (logits)")
    ax[2].set_title("c  Swapping the label", loc="left")
    from matplotlib.lines import Line2D
    ax[2].legend(handles=[Line2D([], [], marker="o", ls="", color=style.TERRACOTTA, ms=3.4, label="label swap"),
                          Line2D([], [], marker="s", ls="", color=M_, ms=3.4, label="position check")],
                 loc="lower left", fontsize=5.8, frameon=False, labelspacing=0.22, borderpad=0.1, handletextpad=0.35)
    style.finish_paper(fig, ax, f"{OUT}/fig_selfanchor_{m}")
    print("wrote", f"{OUT}/fig_selfanchor_{m}.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Q9")
