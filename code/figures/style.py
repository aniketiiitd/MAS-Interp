"""Shared figure style (plan Section 11) with a validated categorical palette.

Categorical slots validated with the dataviz six-check validator on the cream plot surface #F7F3EA
(--pairs all, light): teal #008A7B, terracotta #C8553D, plum #6F56B8 -> ALL CHECKS PASS
(worst CVD dE 10.1, normal-vision dE 21.2, contrast >= 3:1). Slightly more saturated than the plan's muted
teal/plum, which failed the chroma floor. Never more than three categorical hues in one panel.
"""
from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

INK = "#252525"
SLATE = "#4C566A"
MUTED = "#8A8F98"
GRID = "#E2E2E2"   # neutral: the old warm grid was chosen against a cream panel
CREAM = "#F7F3EA"
TEAL = "#008A7B"
TERRACOTTA = "#C8553D"
PLUM = "#6F56B8"
GOLD = "#E9C46A"  # annotation highlight band only, never a data series

DOMAIN_COLOR = {"knowledge_reasoning": TEAL, "values_moral_judgment": PLUM, "safety_refusal": TERRACOTTA}
COMPONENT_COLOR = {"content": TEAL, "provenance": TERRACOTTA, "interaction": PLUM}

# diverging: teal (negative) <- light gray midpoint -> terracotta (positive)
DIVERGING = LinearSegmentedColormap.from_list("cpt_div", ["#006B60", "#7FC2B9", "#EDEAE4", "#E3A08F", "#9E3A26"])
SEQUENTIAL = LinearSegmentedColormap.from_list("cpt_seq", ["#F4EDE9", "#E7B7A9", "#D57E66", "#B04A33", "#7A2E1E"])

FIGSIZE = {"main": (7.2, 3.6), "hero": (7.2, 4.2), "wide": (7.2, 3.0), "casebook": (7.2, 5.0), "tall": (7.2, 5.6)}


def apply():
    from pathlib import Path
    from matplotlib import font_manager
    for f in font_manager.findSystemFonts(fontext="otf"):
        if Path(f).name.startswith("P052-"):  # URW P052 = metric-compatible Palatino clone
            font_manager.fontManager.addfont(f)
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Palatino Linotype", "Palatino", "P052", "TeX Gyre Pagella", "Book Antiqua", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "normal",
        "axes.labelsize": 9.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.edgecolor": INK,
        "axes.linewidth": 0.7,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.7,
        "grid.linestyle": "-",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 360,
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
    })


def finish(fig, axes, path_stem):
    """Save PDF (submission) + PNG (inspection) with the plan's no-overlap settings."""
    import numpy as np

    for ax in np.atleast_1d(axes).flat:
        ax.margins(x=0.04, y=0.08)
        ax.tick_params(length=2.5, pad=3)
        ax.set_axisbelow(True)
    fig.align_labels()
    fig.savefig(f"{path_stem}.pdf", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(f"{path_stem}.png", dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def check_fonts():
    from matplotlib import font_manager

    names = {f.name for f in font_manager.fontManager.ttflist}
    return [n for n in mpl.rcParams["font.serif"] if n in names]


# ---------------------------------------------------------------------------------------------------------------------------
# ICLR paper layout: every figure is a horizontal strip at the template text width (5.5 in), fonts sized for print.
TEXTWIDTH = 5.5


def apply_paper():
    apply()
    mpl.rcParams.update({
        "font.size": 7.2, "axes.titlesize": 7.6, "axes.labelsize": 7.2, "xtick.labelsize": 6.6, "ytick.labelsize": 6.6,
        "legend.fontsize": 6.4, "axes.linewidth": 0.55, "grid.linewidth": 0.4, "lines.linewidth": 1.2, "lines.markersize": 3.5,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "legend.handlelength": 1.4, "legend.handletextpad": 0.4,
        "legend.columnspacing": 0.9, "legend.borderaxespad": 0.2, "axes.titlepad": 3.0, "axes.labelpad": 2.0,
        "figure.constrained_layout.w_pad": 0.01, "figure.constrained_layout.h_pad": 0.01,
        "figure.constrained_layout.wspace": 0.02, "figure.constrained_layout.hspace": 0.02,
    })


def strip(ncols, height=1.9, width_ratios=None):
    fig, axes = plt.subplots(1, ncols, figsize=(TEXTWIDTH, height), constrained_layout=True, squeeze=False,
                             gridspec_kw=dict(width_ratios=width_ratios) if width_ratios else None)
    return fig, axes[0]


def finish_paper(fig, axes, path_stem, width=TEXTWIDTH):
    import numpy as np

    for ax in np.atleast_1d(axes).flat:
        ax.tick_params(length=2.0, pad=1.5)
        ax.set_axisbelow(True)
    fig.align_labels()
    # Land the SAVED file at exactly one text width. bbox_inches="tight" grows past TEXTWIDTH whenever an artist
    # overhangs the canvas (long tick labels, rotated labels), and LaTeX then scales the file down, shrinking the
    # print-size fonts set above; dropping it instead clips those artists. So shrink the canvas until the tight bbox
    # measures TEXTWIDTH, then save tight. Font sizes are absolute, so only the plot area gives way.
    pad = 0.01
    fig.canvas.draw()
    w0 = float(fig.get_size_inches()[0])
    for _ in range(4):
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        over = bb.width + 2 * pad - width
        if abs(over) < 0.004:
            break
        w, h = fig.get_size_inches()
        # Bound the correction: without a floor the loop can keep shrinking a crowded figure until constrained_layout
        # collapses the axes to zero width, which silently produces an unreadable plot rather than an error.
        new_w = min(max(w - over, 0.88 * w0), 1.25 * w0)
        if abs(new_w - w) < 2e-3:
            break
        fig.set_size_inches(new_w, h)
        fig.canvas.draw()
    fig.savefig(f"{path_stem}.pdf", bbox_inches="tight", pad_inches=pad)
    fig.savefig(f"{path_stem}.png", dpi=300, bbox_inches="tight", pad_inches=pad)
    plt.close(fig)
