"""Passing a peer's internal state instead of its words does almost nothing.

  a  every arm at the depth where the transplant works best, as a share of what the peer's written response does.
     The peer's own state barely moves the decision, and an unrelated item's state that happens to argue for the same
     answer moves it almost as much, so the little that does move is not carrying this peer's content.
  b  the same two quantities across depth, so the null is visibly not a bad choice of layer.
  c  what the written response carries. It pushes the reader toward the peer's answer whether that answer is right or
     wrong, so the channel transmits the answer and not its correctness.

The first version of this figure put a six-entry legend inside panel a, on top of six lines that all sit near zero.
Arms are named on the axis here instead, which is what a null result needs.

Usage: python figures/fig_latent.py [MODEL]
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
HOT, COOL = style.TERRACOTTA, style.TEAL

ARMS = [
    ("full_state", "peer's own state", HOT),
    ("same_letter_other_item", "other item, same answer", M_),
    ("deranged_sender", "other episode's peer", M_),
    ("other_position", "elsewhere in prompt", M_),
    ("rotated", "directions scrambled", M_),
    ("random_vector", "random vector", M_),
]


def hdot(ax, y, e, lo, hi, c, m="o", s=20):
    ax.plot([lo, hi], [y, y], color=c, lw=1.2, solid_capstyle="round", zorder=2)
    ax.scatter([e], [y], s=s, marker=m, color=c, zorder=3)


def main(m="Q9", ds="lit1200"):
    style.apply_paper()
    T = pd.read_csv(TABLES / f"lit_latent_{m}_{ds}_transfer.csv")
    U = pd.read_csv(TABLES / f"lit_latent_{m}_{ds}_truth.csv")
    fig, ax = style.strip(3, 2.05, [1.00, 0.78, 0.92])

    fs = T[(T.arm == "full_state") & (T.k == -1)]
    best = int(fs.loc[fs.transfer.idxmax(), "layer"])
    rows = []
    for key, lab, col in ARMS:
        g = T[(T.arm == key) & (T.layer == best)]
        if not len(g):
            g = T[T.arm == key]
            if not len(g):
                continue
            g = g.loc[[g.transfer.abs().idxmax()]]
        r = g.iloc[0]
        rows.append((lab, col, r.transfer, r.lower, r.upper))
    for i, (lab, col, e, lo, hi) in enumerate(rows):
        hdot(ax[0], len(rows) - 1 - i, e, lo, hi, col)
    ax[0].axvline(0, color=S_, lw=0.7)
    ax[0].axvline(1.0, color=HOT, lw=1.0, ls=(0, (3, 2)))
    ax[0].text(0.99, -0.55, "what the words do", fontsize=6.2, color=HOT, ha="right", va="bottom")
    ax[0].set_yticks(range(len(rows)))
    ax[0].set_yticklabels([r[0] for r in rows][::-1])
    ax[0].grid(axis="y", visible=False)
    ax[0].set_ylim(-0.85, len(rows) - 0.4)
    ax[0].set_xlim(-0.10, 1.15)
    ax[0].set_xlabel("share of what the words do")
    ax[0].set_title(f"a  State, not words (L{best})", loc="left")

    for key, lab, col in (("full_state", "the peer's own state", HOT),
                          ("same_letter_other_item", "other item, same answer", M_)):
        g = T[(T.arm == key) & (T.k == -1)] if key == "full_state" else T[T.arm == key]
        g = g.sort_values("layer")
        if not len(g):
            continue
        ax[1].plot(g.layer, g.transfer, color=col, marker="o", ms=3.2, lw=1.3, label=lab)
        ax[1].fill_between(g.layer, g.lower, g.upper, color=col, alpha=0.18, lw=0)
    ax[1].axhline(0, color=S_, lw=0.7)
    ax[1].set_ylim(-0.03, 0.115)
    ax[1].set_xlabel("layer")
    ax[1].set_ylabel("share of the words")
    ax[1].set_title("b  Every depth", loc="left")
    ax[1].legend(loc="upper left", fontsize=6.2, frameon=False, labelspacing=0.25, handletextpad=0.4)

    names = {("text", True): "words, peer right", ("text", False): "words, peer wrong",
             (f"state L{best}", True): "state, peer right", (f"state L{best}", False): "state, peer wrong"}
    order = [("text", True), ("text", False), (f"state L{best}", True), (f"state L{best}", False)]
    for i, key in enumerate(order):
        g = U[(U.channel == key[0]) & (U.sender_correct == key[1])]
        if not len(g):
            continue
        r = g.iloc[0]
        y = len(order) - 1 - i
        hdot(ax[2], y + 0.17, r.toward_sender, r.ts_lower, r.ts_upper, HOT, "o", 18)
        hdot(ax[2], y - 0.17, r.toward_target, r.tt_lower, r.tt_upper, COOL, "s", 18)
    ax[2].axvline(0, color=S_, lw=0.7)
    ax[2].set_yticks(range(len(order)))
    ax[2].set_yticklabels([names[k] for k in order][::-1])
    ax[2].grid(axis="y", visible=False)
    ax[2].set_ylim(-0.95, len(order) - 0.4)
    ax[2].set_xlabel("shift (logits)")
    ax[2].set_title("c  The answer, or the truth?", loc="left")
    from matplotlib.lines import Line2D
    ax[2].legend(handles=[Line2D([], [], marker="o", ls="", color=HOT, ms=3.6, label="toward the peer's answer"),
                          Line2D([], [], marker="s", ls="", color=COOL, ms=3.6, label="toward the right answer")],
                 loc="lower left", fontsize=6.2, frameon=False, labelspacing=0.25, handletextpad=0.4)
    style.finish_paper(fig, ax, f"{OUT}/fig_latent_channel_{m}")
    print(f"wrote {OUT}/fig_latent_channel_{m}.png  (best layer {best})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Q9")
