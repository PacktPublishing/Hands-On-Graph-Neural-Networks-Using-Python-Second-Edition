#!/usr/bin/env python3
"""Compare the numbers two run campaigns printed, chapter by chapter.

The book quotes measured numbers — accuracies, MAEs, F1 scores — in its text
and hard-codes some of them into figure scripts. So the question that matters
before a release is not "did the chapters run" but "did they print the same
numbers as the run the text was written from". This answers that.

Keep a copy of the logs/ directory from the verified campaign, run the
chapters again, then:

    python tools/compare_runs.py /path/to/logs-baseline
    python tools/compare_runs.py /path/to/logs-baseline --only 14 16
    python tools/compare_runs.py /path/to/logs-baseline --context

Warnings on stderr are ignored: they move with library versions without
telling us anything about the results, and a campaign that captured only
stdout would otherwise look different from one that captured both.
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

NUMBER = re.compile(r"-?\d+\.\d+")
NOISE = re.compile(
    r"site-packages|UserWarning|FutureWarning|DeprecationWarning"
    r"|UndefinedMetricWarning|^\s*warn\(|^\s*_warn_prf|^OMP:|^\s*$"
    r"|Downloading|Extracting|Processing\.\.\.|it/s\]|s/it\]")


def numbers(log_path):
    """Every decimal the run printed, in order, minus the noise lines."""
    try:
        text = log_path.read_text(errors="ignore")
    except OSError:
        return None
    out = []
    for line in text.splitlines():
        if NOISE.search(line):
            continue
        out.extend(NUMBER.findall(line))
    return out


def differing_lines(baseline, current):
    """Pairs of lines that hold numbers and do not agree."""
    def keep(p):
        return [l.rstrip() for l in p.read_text(errors="ignore").splitlines()
                if NUMBER.search(l) and not NOISE.search(l)]
    a, b = keep(baseline), keep(current)
    out = []
    for i in range(max(len(a), len(b))):
        x = a[i] if i < len(a) else "(assente)"
        y = b[i] if i < len(b) else "(assente)"
        if x != y:
            out.append((x, y))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline", help="directory holding the verified logs/")
    ap.add_argument("--current", default=str(ROOT / "logs"))
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--context", action="store_true",
                    help="show the lines that differ, not just the count")
    args = ap.parse_args()

    base_dir, cur_dir = Path(args.baseline), Path(args.current)
    if not base_dir.is_dir():
        print(f"baseline non trovata: {base_dir}")
        return 2

    moved, same, absent = [], [], []
    for log in sorted(cur_dir.glob("chapter[0-9][0-9].log")):
        num = log.stem[7:9]
        if args.only and num not in args.only:
            continue
        old = base_dir / log.name
        if not old.exists():
            absent.append((num, "nessun log di riferimento"))
            continue
        a, b = numbers(old), numbers(log)
        if a == b:
            same.append(num)
            continue
        diffs = differing_lines(old, log)
        moved.append((num, len(diffs), diffs))

    print(f"riferimento: {base_dir}")
    print(f"corrente:    {cur_dir}\n")
    if same:
        print(f"IDENTICI ({len(same)}): " + " ".join(f"Chapter{n}" for n in same))
    for num, count, diffs in moved:
        print(f"\nDIVERSI  Chapter{num} — {count} righe con numeri non concordanti")
        for old_line, new_line in diffs[:(None if args.context else 3)]:
            print(f"    prima: {old_line.strip()[:110]}")
            print(f"    ora:   {new_line.strip()[:110]}")
    for num, why in absent:
        print(f"\nSALTATO  Chapter{num} — {why}")

    print(f"\n{len(same)} capitoli riproducono i numeri, {len(moved)} no")
    return 1 if moved else 0


if __name__ == "__main__":
    sys.exit(main())
