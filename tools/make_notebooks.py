#!/usr/bin/env python3
"""Generate a notebook per chapter from its run.py.

run.py stays the single source of truth: the notebook is derived from it, so
the two cannot drift apart. Every chapter marks its sections the same way,

    # =============================================================================
    # PART 2 - Something
    # =============================================================================

and each of those becomes a markdown heading followed by a code cell. The
module docstring becomes the opening markdown cell.

    python tools/make_notebooks.py            # all chapters
    python tools/make_notebooks.py --only 07  # one chapter
    python tools/make_notebooks.py --check    # fail if a notebook is stale

Notebooks are written without outputs; run them in Jupyter to fill those in.
"""

import argparse
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BANNER = re.compile(r"^# ={5,}\s*$")


def split_sections(body):
    """[(title or None, code)] for the code after the module docstring."""
    lines = body.splitlines()
    sections, title, buffer = [], None, []
    i = 0
    while i < len(lines):
        # a section banner is three lines: ====, # TITLE, ====
        if (BANNER.match(lines[i]) and i + 2 < len(lines)
                and BANNER.match(lines[i + 2])):
            if "".join(buffer).strip():
                sections.append((title, "\n".join(buffer).strip("\n")))
            title = lines[i + 1].lstrip("#").strip()
            buffer = []
            i += 3
            continue
        buffer.append(lines[i])
        i += 1
    if "".join(buffer).strip():
        sections.append((title, "\n".join(buffer).strip("\n")))
    return sections


def cell(kind, source):
    base = {"cell_type": kind, "metadata": {},
            "source": source.splitlines(keepends=True)}
    if kind == "code":
        base |= {"execution_count": None, "outputs": []}
    return base


def build(run_py, chapter):
    src = run_py.read_text()
    doc = ast.get_docstring(ast.parse(src)) or ""
    body = src
    if doc:                                   # drop the docstring from the code
        tree = ast.parse(src)
        first = tree.body[0]
        body = "\n".join(src.splitlines()[first.end_lineno:])

    title = doc.strip().splitlines()[0] if doc else chapter
    intro = f"# {title}\n\n" + "\n".join(doc.strip().splitlines()[1:]).strip()
    cells = [cell("markdown", intro.strip())]

    for heading, code in split_sections(body):
        if heading:
            cells.append(cell("markdown", f"## {heading}"))
        cells.append(cell("code", code))

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--check", action="store_true",
                    help="report stale notebooks instead of writing them")
    args = ap.parse_args()

    stale = []
    for chapter_dir in sorted(ROOT.glob("Chapter[0-9][0-9]")):
        num = chapter_dir.name[-2:]
        run_py = chapter_dir / "run.py"
        if not run_py.exists() or (args.only and num not in args.only):
            continue
        nb_path = chapter_dir / f"{chapter_dir.name}.ipynb"
        text = json.dumps(build(run_py, chapter_dir.name), indent=1) + "\n"

        if args.check:
            current = nb_path.read_text() if nb_path.exists() else ""
            if current != text:
                stale.append(nb_path.relative_to(ROOT))
            continue

        nb_path.write_text(text)
        n_code = sum(1 for c in json.loads(text)["cells"] if c["cell_type"] == "code")
        print(f"{nb_path.relative_to(ROOT)}  ({n_code} celle di codice)")

    if args.check:
        for p in stale:
            print(f"STALE  {p} — rigenera con tools/make_notebooks.py")
        print(f"\n{len(stale)} notebook non allineati a run.py")
        return 1 if stale else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
