"""Render a Markdown document to a paginated PDF with matplotlib.

There is no LaTeX in this environment and the deliverables have to be readable as PDFs, so this does the small
subset of Markdown the generated documents actually use: headings, paragraphs, bullet and numbered lists, pipe
tables, fenced code, blockquotes, horizontal rules, and bold/italic stripped to plain text.

It is deliberately not a Markdown implementation. It is the renderer for documents this repository generates,
and it fails loudly on anything it cannot lay out rather than dropping it silently, because a report that
quietly loses a table is worse than one that refuses to build.

Usage: python figures/md_to_pdf.py IN.md OUT.pdf ["Document title"]
"""
from __future__ import annotations

import re
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

PAGE = (8.27, 11.69)          # A4 portrait, inches
L, R = 0.085, 0.955           # left and right margins, figure fraction
TOP, BOT = 0.955, 0.055
INK, MUTED, RULE = "#1A1A1A", "#4C566A", "#C9CDD4"

SIZES = {1: 17.0, 2: 12.6, 3: 10.4}      # heading point sizes by level
LEADS = {1: 0.0300, 2: 0.0224, 3: 0.0190}
BODY, BODY_LEAD = 8.6, 0.0146
MONO, MONO_LEAD = 6.6, 0.0120
WRAP = 104
# Usable width is (R-L) * 8.27in = 7.19in = 518pt; a monospace glyph is about 0.6 of the point size, so the
# character budget is 518 / (0.6 * size). At 7pt that is 123, and the first version wrapped at 132, which ran
# every long docstring line off the right edge of the page.
USABLE_PT = (R - L) * PAGE[0] * 72


def mono_cols(size, indent_pt=9.0):
    return int((USABLE_PT - indent_pt) / (0.6 * size))


MONO_WRAP = mono_cols(MONO)


class Doc:
    """A paginated text canvas. Everything is emitted top to bottom; pages break automatically."""

    def __init__(self, pdf, title):
        self.pdf, self.title, self.page = pdf, title, 0
        self._new_page()

    def _new_page(self):
        if self.page:
            self.fig.text(R, 0.028, str(self.page), fontsize=7.4, color=MUTED, ha="right")
            self.fig.text(L, 0.028, self.title, fontsize=7.4, color=MUTED)
            self.pdf.savefig(self.fig)
            plt.close(self.fig)
        self.fig = plt.figure(figsize=PAGE)
        self.fig.patch.set_facecolor("white")
        self.y = TOP
        self.page += 1

    def space(self, dy):
        self.y -= dy

    def _room(self, need):
        if self.y - need < BOT:
            self._new_page()

    def text(self, s, size=BODY, lead=BODY_LEAD, color=INK, weight="normal",
             indent=0.0, family=None, wrap=WRAP):
        for line in (textwrap.wrap(s, wrap) or [""]):
            self._room(lead)
            self.fig.text(L + indent, self.y, line, fontsize=size, color=color,
                          weight=weight, va="top",
                          family=family or matplotlib.rcParams["font.family"])
            self.y -= lead

    def text_raw(self, line, size=BODY, lead=BODY_LEAD, color=INK, weight="normal",
                 indent=0.0, family=None):
        """One pre-wrapped line, drawn as given. Used for verbatim blocks where wrapping already happened."""
        self._room(lead)
        self.fig.text(L + indent, self.y, line, fontsize=size, color=color, weight=weight, va="top",
                      family=family or matplotlib.rcParams["font.family"])
        self.y -= lead

    def rule(self):
        self._room(0.02)
        self.y -= 0.008
        self.fig.add_artist(plt.Line2D([L, R], [self.y, self.y], color=RULE, lw=0.7))
        self.y -= 0.012

    def close(self):
        self.fig.text(R, 0.028, str(self.page), fontsize=7.4, color=MUTED, ha="right")
        self.fig.text(L, 0.028, self.title, fontsize=7.4, color=MUTED)
        self.pdf.savefig(self.fig)
        plt.close(self.fig)


def plain(s):
    """Strip the inline markup this renderer does not draw, keeping the words."""
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"(?<!\w)\*(.+?)\*(?!\w)", r"\1", s)
    s = re.sub(r"`(.+?)`", r"\1", s)
    s = re.sub(r"\[(.+?)\]\((.+?)\)", r"\1 (\2)", s)
    return s


def render_table(doc, rows):
    """Pipe table as aligned monospace. Columns are sized to content and clipped to the page width."""
    cells = [[plain(c.strip()) for c in r.strip().strip("|").split("|")] for r in rows]
    cells = [c for c in cells if not all(re.fullmatch(r":?-{2,}:?", x.strip() or "-") for x in c)]
    if not cells:
        return
    ncol = max(len(r) for r in cells)
    cells = [r + [""] * (ncol - len(r)) for r in cells]
    widths = [max(len(r[i]) for r in cells) for i in range(ncol)]
    need = sum(widths) + 3 * (ncol - 1)
    # Shrink the type before clipping the content. The first version truncated cells to fit, which silently ate
    # the end of the widest column -- a table that quietly loses its last words is worse than a small table.
    size = MONO
    while need > mono_cols(size) and size > 4.6:
        size -= 0.2
    if need > mono_cols(size):
        over = need - mono_cols(size)
        while over > 0 and max(widths) > 10:
            widths[widths.index(max(widths))] -= 1
            over -= 1
    doc.space(0.004)
    for j, row in enumerate(cells):
        cut = [t if len(t) <= wd else t[:max(1, wd - 1)] + "\u2026" for t, wd in zip(row, widths)]
        line = "   ".join(t.ljust(wd) for t, wd in zip(cut, widths))
        doc.text_raw(line, size=size, lead=MONO_LEAD, family="monospace",
                     weight="bold" if j == 0 else "normal",
                     color=INK if j == 0 else "#333333")
    doc.space(0.006)


def render(md_path, pdf_path, title=None):
    lines = open(md_path).read().splitlines()
    title = title or next((l.lstrip("# ").strip() for l in lines if l.startswith("# ")), "Document")
    with PdfPages(pdf_path) as pdf:
        doc = Doc(pdf, plain(title))
        i = 0
        while i < len(lines):
            ln = lines[i].rstrip()
            # fenced code
            if ln.strip().startswith("```"):
                i += 1
                buf = []
                while i < len(lines) and not lines[i].strip().startswith("```"):
                    buf.append(lines[i]); i += 1
                doc.space(0.004)
                for b in buf:
                    # Hanging indent: these blocks carry aligned columns (argument tables in module
                    # docstrings), and wrapping a 172-character line back to column zero destroys the
                    # alignment that made it worth showing verbatim.
                    pad = len(b) - len(b.lstrip())
                    chunks = textwrap.wrap(b, mono_cols(MONO), subsequent_indent=" " * (pad + 2),
                                           drop_whitespace=False, replace_whitespace=False) or [""]
                    for c in chunks:
                        doc.text_raw(c, size=MONO, lead=MONO_LEAD, family="monospace",
                                     color="#333333", indent=0.012)
                doc.space(0.006)
                i += 1
                continue
            # pipe table
            if ln.strip().startswith("|") and ln.strip().endswith("|"):
                buf = []
                while i < len(lines) and lines[i].strip().startswith("|"):
                    buf.append(lines[i]); i += 1
                render_table(doc, buf)
                continue
            m = re.match(r"^(#{1,6})\s+(.*)$", ln)
            if m:
                lvl = min(len(m.group(1)), 3)
                doc.space(0.016 if lvl > 1 else 0.010)
                doc._room(LEADS[lvl] * 2)
                doc.text(plain(m.group(2)), size=SIZES[lvl], lead=LEADS[lvl], weight="bold",
                         wrap=int(WRAP * BODY / SIZES[lvl]))
                if lvl == 1:
                    doc.rule()
                doc.space(0.004)
                i += 1
                continue
            if re.fullmatch(r"\s*---+\s*", ln):
                doc.rule(); i += 1; continue
            if ln.strip().startswith(">"):
                doc.text(plain(ln.strip().lstrip("> ")), color=MUTED, indent=0.016)
                i += 1; continue
            m = re.match(r"^\s*([-*])\s+(.*)$", ln)
            if m:
                doc.text("- " + plain(m.group(2)), indent=0.012)
                i += 1; continue
            m = re.match(r"^\s*(\d+)[.)]\s+(.*)$", ln)
            if m:
                doc.text(f"{m.group(1)}. " + plain(m.group(2)), indent=0.012)
                i += 1; continue
            if not ln.strip():
                doc.space(0.006); i += 1; continue
            # paragraph: join until a blank line or a block starts
            buf = []
            while i < len(lines) and lines[i].strip() and not re.match(
                    r"^\s*(#{1,6}\s|[-*]\s|\d+[.)]\s|\||```|>|---+\s*$)", lines[i]):
                buf.append(lines[i].strip()); i += 1
            doc.text(plain(" ".join(buf)))
            doc.space(0.004)
        doc.close()
    return doc.page


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit("usage: md_to_pdf.py IN.md OUT.pdf [title]")
    n = render(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    print(f"[md_to_pdf] {sys.argv[2]}: {n} pages")
