#!/usr/bin/env python3
"""Check that every chapter's requirements agree with the pinned root set.

The root requirements.txt is the environment the book was verified against.
A chapter's requirements.txt can be installed on its own, into an environment
of its own, so it pins the same versions rather than a floor: a floor would
promise that the chapter still runs on whatever is newest years from now,
which nobody has tried. This reports:

  MISMATCH  a chapter states a version the root file does not pin
  MISSING   a chapter requires a package the root file does not pin
  WARNING   a chapter has run.py but no requirements.txt

    python tools/check_requirements.py
"""

import re
import sys
from pathlib import Path

try:
    from packaging.requirements import Requirement
except ImportError:
    sys.exit("pip install packaging")

ROOT = Path(__file__).resolve().parent.parent


def normalise(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def parse(path):
    out = {}
    for line in path.read_text().splitlines():
        line = line.split("#")[0].strip()
        if not line or line.startswith("-"):
            continue
        try:
            req = Requirement(line)
        except Exception:
            continue
        out[normalise(req.name)] = req
    return out


def main():
    root = parse(ROOT / "requirements.txt")
    problems = warnings = 0

    for path in sorted(ROOT.glob("Chapter[0-9][0-9]/requirements.txt")):
        chapter = path.parent.name
        for name, req in parse(path).items():
            if name not in root:
                print(f"MISSING   {chapter}: {req} is not pinned in requirements.txt")
                problems += 1
            elif str(req) != str(root[name]):
                print(f"MISMATCH  {chapter}: states {req}, root pins {root[name]}")
                problems += 1

    missing = [d.name for d in sorted(ROOT.glob("Chapter[0-9][0-9]"))
               if (d / "run.py").exists() and not (d / "requirements.txt").exists()]
    for chapter in missing:
        print(f"WARNING   {chapter}: has run.py but no requirements.txt")
        warnings += 1

    print(f"\n{problems} problemi, {warnings} avvisi")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
