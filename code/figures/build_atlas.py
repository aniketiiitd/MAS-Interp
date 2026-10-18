"""Render every figure that could go in the paper, one per page, with its caption and its provenance.

`build_pdf.py` renders the six figures the paper ships. This renders those plus the supplementary plots, and
labels the superseded ones as superseded instead of quietly leaving them in a folder where they look usable.
Three of the PNGs in the paper directory are earlier versions of figures that have since been rebuilt on more
models or a different concept set, and handing those over without a warning is how a retracted result gets back
into a paper.

Out: outputs/figures/MAD_Interp_figure_atlas.pdf
"""
from __future__ import annotations

import glob
import os
import re
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "outputs/figures/paper"
PAGE = (8.27, 11.69)
L, INK, MUTED = 0.075, "#1A1A1A", "#4C566A"

# Figures that are earlier versions of something still in the paper. Each says what replaced it and why.
SUPERSEDED = {
    "V3_traj_domain_Q9": "Replaced by V3_traj_domain. That version was Qwen only and its lead panel plotted "
                         "pressure_from_numbers, a concept Gemma cannot validate; the current figure is three "
                         "models on the 99 concepts all three validate.",
    "fig_turns_Q9": "Replaced by fig_turns_three. Qwen only.",
    "fig_turns_Q9_L31": "Replaced by fig_turns_three. Two models; Gemma's twelve-turn run landed afterwards.",
}
# Plots that are not in the six but are current and usable, with what they show.
SUPPLEMENTARY = {
    "fig_routing": "Attention knockout by layer: how much of a peer's effect each cumulative mask removes, per "
                   "model. The layer-resolved version of the headline number in claim 3.",
    "fig_latent_cross": "Cross-model latent transfer: mapping a sender's decision state into a receiver.",
    "fig_latent_cross_controls": "The controls that turn the transfer result into a null: same letter different "
                                 "item, mean sender state, map fitted on shuffled pairs.",
    "fig_latent_cross_depths": "Transfer by depth, with reconstruction fidelity against transfer strength.",
    "fig_concept_G4": "Gemma concept-probe validation: out-of-fold AUROC by layer and concept.",
    "fig_madbench_map_Q9": "Benchmark map: where items sit by solo accuracy and answer entropy, by domain.",
}


def hero_captions():
    txt = (FIG / "HERO_CAPTIONS.md").read_text()
    blocks = re.split(r"^## Figure (\d+): `([\w./-]+?)\.\{pdf,png\}` \(script `([^`]+)`\)\s*$", txt, flags=re.M)
    out = []
    for i in range(1, len(blocks), 4):
        num, stem, script, body = blocks[i], blocks[i + 1], blocks[i + 2], blocks[i + 3]
        body = re.split(r"Numbers on the figure:", body)[0]
        m = re.match(r"\s*\*\*(.+?)\*\*\s*(.*)$", body.strip(), re.S)
        title, rest = (m.group(1), m.group(2)) if m else (stem, body)
        out.append((stem, f"Figure {num}", script, " ".join(title.split()), " ".join(rest.split())))
    return out


def page(pdf, stem, kicker, title, caption, script, note=None):
    path = FIG / f"{stem}.png"
    if not os.path.exists(path):
        return False
    fig = plt.figure(figsize=PAGE)
    fig.patch.set_facecolor("white")
    fig.text(L, 0.962, kicker, fontsize=8.6, color=MUTED, weight="bold")
    y = 0.940
    for line in textwrap.wrap(title, 74):
        fig.text(L, y, line, fontsize=13.2, weight="bold", color=INK)
        y -= 0.0225
    img = mpimg.imread(path)
    h, w = img.shape[0], img.shape[1]
    fw = 1 - 2 * L
    fh = min(fw * (h / w) * (PAGE[0] / PAGE[1]), 0.46)
    top = y - 0.012
    ax = fig.add_axes([L, top - fh, fw, fh])
    ax.imshow(img)
    ax.axis("off")
    y = top - fh - 0.028
    if note:
        for line in textwrap.wrap(note, 108):
            fig.text(L, y, line, fontsize=8.4, color="#A03020", va="top", weight="bold")
            y -= 0.0140
        y -= 0.006
    for line in textwrap.wrap(caption, 112):
        fig.text(L, y, line, fontsize=8.2, color="#333333", va="top")
        y -= 0.0134
    y -= 0.010
    fig.text(L, y, f"file: outputs/figures/paper/{stem}.png   |   script: {script}",
             fontsize=7.2, color=MUTED, va="top", family="monospace")
    pdf.savefig(fig)
    plt.close(fig)
    return True


def script_for(stem):
    for f in sorted(glob.glob(str(ROOT / "code/figures/*.py"))):
        if "_retired" in f:
            continue
        if stem in open(f).read():
            return "figures/" + os.path.basename(f)
    return "(not referenced by a current figure script)"


def main():
    out = ROOT / "outputs/figures/MAD_Interp_figure_atlas.pdf"
    heroes = hero_captions()
    shipped = {s for s, *_ in heroes}
    n = 0
    with PdfPages(out) as pdf:
        for stem, num, script, title, cap in heroes:
            n += page(pdf, stem, f"{num}  ·  IN THE PAPER", title, cap, script)
        for stem, why in SUPPLEMENTARY.items():
            n += page(pdf, stem, "SUPPLEMENTARY  ·  current, not in the six", stem, why, script_for(stem))
        for stem, why in SUPERSEDED.items():
            n += page(pdf, stem, "SUPERSEDED  ·  do not use", stem,
                      "Kept only so that an older copy found elsewhere can be identified as out of date.",
                      script_for(stem), note="SUPERSEDED: " + why)
    missing = [s for s in glob.glob(str(FIG / "*.png"))
               if os.path.basename(s)[:-4] not in shipped | set(SUPPLEMENTARY) | set(SUPERSEDED)]
    if missing:
        print(f"[atlas] WARNING: {len(missing)} PNG(s) in the paper directory are in no tier and were skipped:")
        for m in missing:
            print("   ", os.path.basename(m))
    print(f"[done] {out}: {n} figures "
          f"({len(heroes)} shipped, {len(SUPPLEMENTARY)} supplementary, {len(SUPERSEDED)} superseded)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
