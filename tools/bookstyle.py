#!/usr/bin/env python3
"""Draw every figure at the size it is printed, so nothing is scaled down.

Packt sets the body text in a column COL_WIDTH inches wide. A figure handed
over wider than that is reduced to fit, and every label in it shrinks by the
same factor: a figure drawn 13 inches wide and placed at 5.5 arrives at 0.42x,
so its 9 pt labels reach the page as 4 pt. That, and not the choice of font or
the resolution, is why the reviewers rejected them for "Readability of text" —
the same figures were already being supplied at over 300 dpi.

Draw at COL_WIDTH and the scale factor is 1.0 — the point size asked for here
is the point size the reader gets.

    from bookstyle import apply, figure, save, G0, G3

    apply()
    fig, ax = figure(aspect=0.42)        # COL_WIDTH x (COL_WIDTH * 0.42)
    ax.plot(...)
    save(fig, 'fig13_1_mpnn.png')

A figure that genuinely cannot work at column width takes width=PAGE_WIDTH and
is placed full-width on the page; that costs 15% more room, not 2x, so it is
not an escape from re-laying-out a wide diagram.

Run this file to audit the repository:

    python tools/bookstyle.py            # report figures that will not survive
    python tools/bookstyle.py --pngs     # also check the resolution of the PNGs
"""

import argparse
import re
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# The print geometry. These four numbers decide everything else; if Packt
# confirms a different column width, change COL_WIDTH and re-run the audit.
# ---------------------------------------------------------------------------
# Measured from the manuscripts, not guessed: their template is US Letter with
# 1.1in margins, a 6.30in column, and the figures are placed in it at 5.50in
# (Chapter 9), 5.80in (Chapters 10 and 19) and 6.00in (Chapter 20). Authoring
# at the narrowest of those means no placement ever reduces a figure.
COL_WIDTH = 5.5      # inches: the width a figure is placed at, so its own width
PAGE_WIDTH = 6.3     # inches: the manuscript column, for a figure that needs it
# 300 dpi is the print floor and is enough: the rejected figures of Chapters 10,
# 19 and 20 sat at 319-463 dpi and were still rejected, for their text size.
DPI = 300
MIN_PT = 8.0         # smallest point size that survives printing

# The grayscale palette the chapters share, darkest first. The book is printed
# without colour, so figures are authored that way rather than converted.
FONT = "DejaVu Sans"
BG = "white"
G0, G1, G2, G3 = "#111111", "#333333", "#555555", "#777777"
G4, G5, G6 = "#999999", "#BBBBBB", "#DDDDDD"
GREYS = [G0, G1, G2, G3, G4, G5, G6]


def apply():
    """Set the rcParams every figure in the book starts from."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.family': FONT,
        'font.size': 9.5,          # the body of a label
        'axes.titlesize': 10.5,
        'axes.labelsize': 9.5,
        'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5,
        'legend.fontsize': 8.5,
        'figure.titlesize': 11,
        'figure.dpi': DPI,
        'savefig.dpi': DPI,
        'figure.facecolor': BG,
        'savefig.facecolor': BG,
        'axes.edgecolor': G3,
        'axes.labelcolor': G0,
        'text.color': G0,
        'xtick.color': G1,
        'ytick.color': G1,
        'grid.color': G6,
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'lines.linewidth': 1.4,
        'patch.linewidth': 1.0,
        # constrained_layout gives the axes room to place ticks and labels
        # without overlapping, which is the other thing the reviewers flagged.
        'figure.constrained_layout.use': True,
        'figure.constrained_layout.h_pad': 0.04,
        'figure.constrained_layout.w_pad': 0.04,
    })
    return plt


def figure(aspect=0.6, width=COL_WIDTH, **kwargs):
    """plt.subplots at the printed width. `aspect` is height / width."""
    import matplotlib.pyplot as plt
    return plt.subplots(figsize=(width, width * aspect), **kwargs)


def save(fig, path, dpi=DPI):
    """Write the figure, without bbox_inches='tight'.

    'tight' crops to the ink and so silently changes the width the figure was
    authored at — which is the whole point of drawing at print size. Let
    constrained_layout do the fitting instead.
    """
    import matplotlib.pyplot as plt
    fig.savefig(path, dpi=dpi, facecolor=BG, edgecolor='none')
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------
FIGSIZE = re.compile(r'figsize=\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)')
FONTSIZE = re.compile(r'(?:fontsize|font_size|fs)\s*=\s*(\d+(?:\.\d+)?)')
USES_BOOKSTYLE = re.compile(r'^\s*(?:from|import)\s+bookstyle', re.M)


def scripts():
    """Every script in the repository that writes a figure."""
    out = []
    for chapter in sorted(ROOT.glob("Chapter[0-9][0-9]")):
        out.extend(sorted(chapter.glob("figures/generate_figures*.py")))
        run_py = chapter / "run.py"
        if run_py.exists():
            out.append(run_py)
    return out


def audit_script(path):
    """[(line, width, min_pt_declared, effective_pt)] for each figure drawn.

    The width is the one declared in figsize. Scripts that save with
    bbox_inches='tight' hand over something narrower, because the crop removes
    the margins, so the printed point size is better than reported here — the
    13-inch figure 13.1, for one, reaches the page 8.6 inches wide. Treat the
    effective size as the worst case, not the measurement.
    """
    src = path.read_text()
    if USES_BOOKSTYLE.search(src):
        return []                     # already authored at print width

    rows = []
    hits = list(FIGSIZE.finditer(src))
    for i, m in enumerate(hits):
        width = float(m.group(1))
        end = hits[i + 1].start() if i + 1 < len(hits) else len(src)
        sizes = [float(s) for s in FONTSIZE.findall(src[m.end():end])]
        if not sizes:
            continue
        smallest = min(sizes)
        scale = min(1.0, COL_WIDTH / width)
        rows.append((src[:m.start()].count('\n') + 1, width,
                     smallest, smallest * scale))
    return rows


def png_dpi(path):
    """Effective dpi if the PNG is printed at column width."""
    with open(path, 'rb') as handle:
        header = handle.read(33)
    if header[:8] != b'\x89PNG\r\n\x1a\n':
        return None
    width, _ = struct.unpack('>II', header[16:24])
    return width / COL_WIDTH


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--pngs', action='store_true',
                    help='also report PNGs below %d dpi at column width' % DPI)
    args = ap.parse_args()

    print(f"Column width {COL_WIDTH}in, minimum {MIN_PT} pt, {DPI} dpi")
    print("Widths come from figsize; a bbox_inches='tight' crop makes the "
          "real figure narrower,\nso these point sizes are the worst case.\n")

    failing = 0
    for path in scripts():
        rows = [r for r in audit_script(path) if r[3] < MIN_PT]
        if not rows:
            continue
        print(f"{path.relative_to(ROOT)}")
        for line, width, declared, effective in rows:
            print(f"  line {line:>4}  {width:>5.1f}in wide  "
                  f"{declared:>4.1f} pt declared -> {effective:>4.1f} pt printed")
            failing += 1
        print()

    print(f"{failing} figure(s) whose text will not be readable in print")

    if args.pngs:
        low = []
        for chapter in sorted(ROOT.glob("Chapter[0-9][0-9]")):
            for png in sorted(chapter.glob("figures/*.png")):
                dpi = png_dpi(png)
                if dpi is not None and dpi < DPI:
                    low.append((png.relative_to(ROOT), dpi))
        print(f"\n{len(low)} PNG(s) below {DPI} dpi at column width")
        for name, dpi in low:
            print(f"  {dpi:>5.0f} dpi  {name}")

    return 1 if failing else 0


if __name__ == '__main__':
    sys.exit(main())
