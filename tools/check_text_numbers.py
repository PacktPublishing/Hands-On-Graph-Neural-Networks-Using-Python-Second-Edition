#!/usr/bin/env python3
"""Shortlist the numbers a chapter's text claims that its run did not print.

The chapters print measured numbers; the text quotes them. Between a code
rework and a copy-editing pass those two drift apart silently, and the drift is
invisible to every other check in this repository — the notebooks, the READMEs
and the figure indexes are all derived from the code, so they agree with each
other while disagreeing with the book.

This reads the chapter's manuscript, pulls out every number that looks like a
measurement, and reports the ones that do not appear anywhere in that chapter's
run log.

    python tools/check_text_numbers.py 10
    python tools/check_text_numbers.py 10 14 16
    python tools/check_text_numbers.py --all

A flagged number is a candidate, not a verdict. The text legitimately cites
numbers that never reach a log: published benchmark results, parameter counts,
dataset sizes, years. Read the sentence and decide. What matters is that a
number the chapter presents as *its own* result should be findable in its own
output, and this is the list of those that are not.

Needs python-docx. The repository's .venv has it:

    .venv/bin/python tools/check_text_numbers.py 10
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# A measurement, as the book writes them: 0.38, 24.37%, 76.10%, 12.2059.
MEASUREMENT = re.compile(r"(?<![\w.])(\d{1,3}\.\d{1,4})\s*%?(?![\w.])")

# Sentences that quote someone else's number rather than the chapter's own.
CITED = re.compile(
    r"published|benchmark|paper|literature|reported in|state of the art"
    r"|leaderboard|et al|Table \d|billion|million|parameters", re.I)


def text_of(docx_path):
    try:
        import docx
    except ImportError:
        sys.exit("python-docx non installato — usa .venv/bin/python")
    document = docx.Document(str(docx_path))
    paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                paragraphs.append("  ".join(cells))
    return paragraphs


def manuscript(chapter_dir):
    found = sorted(chapter_dir.glob("*.docx"))
    return found[0] if found else None


def log_numbers(num):
    log = ROOT / "logs" / f"chapter{num}.log"
    if not log.exists():
        return None
    return set(MEASUREMENT.findall(log.read_text(errors="ignore")))


def check(num, show_cited):
    chapter_dir = ROOT / f"Chapter{num}"
    doc = manuscript(chapter_dir)
    if doc is None:
        return f"Chapter{num}: nessun .docx nel capitolo", None
    printed = log_numbers(num)
    if printed is None:
        return f"Chapter{num}: nessun logs/chapter{num}.log — eseguilo prima", None

    missing = []
    for para in text_of(doc):
        if not show_cited and CITED.search(para):
            continue
        for m in MEASUREMENT.finditer(para):
            value = m.group(1)
            if value in printed or value in {"0.0", "1.0"}:
                continue
            before = para[max(0, m.start() - 14):m.start()]
            # a hyperparameter in a code snippet, not a claim: dropout=0.2
            if re.search(r"[=(\[]\s*$", before):
                continue
            # a cross-reference, not a measurement: "Figure 10.3", "Section 2.1"
            if re.search(r"(Figure|Fig\.|Table|Section|Chapter|Equation)\s*$",
                         before, re.I):
                continue
            missing.append((value, para))
    return None, (doc, printed, missing)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chapters", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--cited", action="store_true",
                    help="keep sentences that look like they quote someone else")
    args = ap.parse_args()

    if args.all:
        nums = sorted(d.name[-2:] for d in ROOT.glob("Chapter[0-9][0-9]")
                      if list(d.glob("*.docx")))
    else:
        nums = [n.zfill(2) for n in (args.chapters or [])]
    if not nums:
        return ap.print_help() or 0

    flagged = 0
    for num in nums:
        problem, result = check(num, args.cited)
        if problem:
            print(f"\n{problem}")
            continue
        doc, printed, missing = result
        print(f"\n=== Chapter{num} — {doc.name}")
        print(f"    {len(printed)} valori distinti nel log")
        if not missing:
            print("    ogni numero del testo si ritrova nell'output")
            continue
        flagged += len(missing)
        print(f"    {len(missing)} numeri del testo NON presenti nell'output:\n")
        for value, para in missing:
            print(f"    {value:>9}  {para[:150]}")

    print(f"\n{flagged} numeri da controllare a mano")
    return 1 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())
