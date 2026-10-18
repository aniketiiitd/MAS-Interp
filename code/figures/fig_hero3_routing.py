"""Hero figure 3: cutting every attention path to a peer's message does not cut off its influence.

Visual form: one comparison, two columns. The same intervention is run on two architectures and the result is a single
height against a single reference line, so the claim is legible before any label is read. 100% is what deleting the
peer's message does. Llama-3.1-8B, where all 32 layers are attention, goes past that line: nothing is left that the
mask cannot reach. Qwen3.5-9B, where only 8 of 32 layers are attention and the other 24 are Gated-DeltaNet, stops
short, and the gap up to the line is influence the mask never touched. Under each bar the model's 32 layers are drawn
as a strip, which is the reason the two bars differ.

The earlier version put the layer strips in their own column beside the bars, so the reader had to carry four things
across the page before the point arrived. Here each strip sits under the bar it explains.

Reads outputs/tables/lit_routing_{Q9,L31}_summary.csv.
Usage: python figures/fig_hero3_routing.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/figures"))
import style  # noqa: E402
from style import plt  # noqa: E402

TABLES = ROOT / "outputs/tables"
OUT = ROOT / "outputs/figures/paper"
ARM_DEC = "peer keys blocked at every attention layer (decision query)"
ARM_ALL = "peer keys blocked for every later query at every attention layer"
ARM_OWN = "control: own-response keys of matched length"
ARM_PRE = "control: peer keys blocked for the prefill tokens, not the decision token"
ATTN = {"L31": list(range(32)), "Q9": [3, 7, 11, 15, 19, 23, 27, 31]}
MNAME = {"L31": "Llama-3.1-8B", "Q9": "Qwen3.5-9B"}
DARK, LIGHT, HOT = style.SLATE, "#A9B1C1", style.TERRACOTTA
HATCHBG = "#F6E4DE"

W, H = style.TEXTWIDTH, 3.34
YB, YT = 1.55, 2.76          # the bar lane, in inches
PMAX = 150.0                 # percent at the top of the lane
COLS = {"L31": 2.05, "Q9": 4.10}
BW = 0.78


def py(p):
    return YB + (YT - YB) * p / PMAX


def load(m):
    return pd.read_csv(TABLES / f"lit_routing_{m}_summary.csv").set_index("arm")


def strip(ax, m, xc, y, w, h):
    """The model's 32 layers, solid where an attention mask has something to act on."""
    n = 32
    pitch = w / n
    bw = pitch * 0.78
    x0 = xc - w / 2
    for i in range(n):
        x = x0 + i * pitch
        if i in ATTN[m]:
            ax.add_patch(Rectangle((x, y), bw, h, facecolor=DARK, edgecolor="none", zorder=4))
        else:
            ax.add_patch(Rectangle((x, y), bw, h, facecolor=HATCHBG, edgecolor=HOT, lw=0.35, hatch="///", zorder=4))


def main():
    style.apply_paper()
    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.set_facecolor("white")

    # No title inside the figure: the paper's caption carries it, and printing it twice was just noise. What stays is
    # the line that says what the axis means.
    ax.text(0.10, H - 0.10, "the peer's message is blocked at every attention layer; 100% is deleting it outright",
            ha="left", va="top", fontsize=6.6, color=style.SLATE)

    ax.plot([0.95, W - 0.10], [py(100), py(100)], color=style.INK, lw=0.8, ls=(0, (2.5, 2)), zorder=3)
    ax.text(0.90, py(100), "deleting the\nmessage: 100%", ha="right", va="center", fontsize=6.2, color=style.INK,
            linespacing=1.15)

    Q, L = load("Q9"), load("L31")
    SAYS = {"L31": "all 32 layers are attention, so the mask reaches\nevery one, and it removes more than deleting did",
            "Q9": "8 of 32 are attention; the other 24 are recurrent,\nno mask reaches them, and a quarter survives"}
    for m, xc in COLS.items():
        S = Q if m == "Q9" else L
        a = 100 * S.loc[ARM_DEC, "share"]
        b, bl, bh = (100 * S.loc[ARM_ALL, c] for c in ("share", "share_lo", "share_hi"))
        ax.add_patch(Rectangle((xc - BW / 2, py(bl)), BW, py(min(bh, PMAX)) - py(bl), facecolor=LIGHT,
                               edgecolor="none", alpha=0.42, zorder=2))
        ax.add_patch(Rectangle((xc - BW / 2, py(0)), BW, py(min(b, PMAX)) - py(0), facecolor=LIGHT,
                               edgecolor="none", zorder=3))
        ax.add_patch(Rectangle((xc - BW / 2, py(0)), BW, py(a) - py(0), facecolor=DARK, edgecolor="none", zorder=4))
        if b < 100:
            ax.add_patch(Rectangle((xc - BW / 2, py(b)), BW, py(100) - py(b), facecolor=HATCHBG, edgecolor=HOT,
                                   lw=0.7, hatch="///", zorder=4))
        # A bar that stops short of the line still puts its number above the line, not inside the hatched gap it
        # leaves behind, which is where Qwen's 74% was landing.
        ax.text(xc, py(max(b, 100.0)) + 0.07, f"{b:.0f}%", ha="center", va="bottom", fontsize=13.5,
                color=style.INK, fontweight="bold")
        ax.text(xc, py(a / 2), f"{a:.0f}%", ha="center", va="center", fontsize=6.4, color="white",
                fontweight="bold", zorder=5)
        ax.text(xc, YB - 0.09, MNAME[m], ha="center", va="top", fontsize=7.6, color=style.INK, fontweight="bold")
        strip(ax, m, xc, 1.14, BW + 0.42, 0.12)
        ax.text(xc, 1.06, SAYS[m], ha="center", va="top", fontsize=6.0, color=style.SLATE, linespacing=1.25)

    left = 100 - 100 * Q.loc[ARM_ALL, "share"]
    ll, lh = 100 - 100 * Q.loc[ARM_ALL, "share_hi"], 100 - 100 * Q.loc[ARM_ALL, "share_lo"]
    ax.text(COLS["Q9"] + BW / 2 + 0.10, py(100 - left / 2), f"{left:.0f}% survives\nthe mask [{ll:.0f}\u2013{lh:.0f}%]",
            ha="left", va="center", fontsize=6.6, color=HOT, fontweight="bold", linespacing=1.25)

    # one horizontal key row, then the checks on the last line: stacking these two blocks was what collided before
    x = 0.10
    for c, hatch, lab in ((DARK, None, "deciding token blocked"), (LIGHT, None, "all later tokens blocked too"),
                          (HATCHBG, "///", "never reached by the mask")):
        ax.add_patch(Rectangle((x, 0.50), 0.10, 0.084, facecolor=c, edgecolor=HOT if hatch else "none",
                               lw=0.5, hatch=hatch, zorder=3))
        ax.text(x + 0.15, 0.542, lab, ha="left", va="center", fontsize=5.9, color=style.INK)
        x += 0.28 + 0.052 * len(lab)
    ax.text(0.10, 0.34, "Checks (Qwen / Llama): blocking the agent's OWN text of matched length instead removes "
            f"{100 * Q.loc[ARM_OWN, 'share']:.0f}% / {100 * L.loc[ARM_OWN, 'share']:.0f}%, making it more\n"
            "persuaded, not less; blocking the peer for the prompt tokens but not the deciding token removes "
            f"{100 * Q.loc[ARM_PRE, 'share']:.0f}% / {100 * L.loc[ARM_PRE, 'share']:.0f}%.".replace("-", "\u2212"),
            ha="left", va="top", fontsize=5.7, color=style.SLATE, linespacing=1.3)

    style.finish_paper(fig, ax, str(OUT / "fig_hero3_routing"))
    print("wrote", OUT / "fig_hero3_routing.png")


if __name__ == "__main__":
    main()
