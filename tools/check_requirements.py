#!/usr/bin/env python3
"""Check that every chapter's requirements agree with the pinned root set.

The root requirements.txt is the environment the book was verified against.
Each chapter states the minimum it needs. This reports:

  CONFLICT  a chapter asks for something the pinned version does not satisfy
  MISSING   a chapter requires a package the root file does not pin
  WARNING   a chapter pins an exact version, or needs a newer minor than the
            rest of the book, which is worth a note in its README

    python tools/check_requirements.py
"""

import re
import sys
from pathlib import Path

try:
    from packaging.requirements import Requirement
    from packaging.version import Version
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


def pinned_version(req):
    for spec in req.specifier:
        if spec.operator in ("==", "==="):
            return Version(spec.version.replace("+pt25", ""))
    return None            # e.g. a git URL: no comparable version


def main():
    root = parse(ROOT / "requirements.txt")
    problems = warnings = 0

    for path in sorted(ROOT.glob("Chapter[0-9][0-9]/requirements.txt")):
        chapter = path.parent.name
        for name, req in parse(path).items():
            if name not in root:
                print(f"MISSING   {chapter}: {req} is not pinned in requirements.txt")
                problems += 1
                continue
            version = pinned_version(root[name])
            if version is None:                    # git pin, cannot compare
                continue
            if not req.specifier.contains(version, prereleases=True):
                print(f"CONFLICT  {chapter}: needs {req}, root pins {version}")
                problems += 1
                continue
            for spec in req.specifier:
                if spec.operator == "==":
                    print(f"WARNING   {chapter}: pins {req} exactly — note it in "
                          f"the chapter README")
                    warnings += 1

    missing = [d.name for d in sorted(ROOT.glob("Chapter[0-9][0-9]"))
               if (d / "run.py").exists() and not (d / "requirements.txt").exists()]
    for chapter in missing:
        print(f"WARNING   {chapter}: has run.py but no requirements.txt")
        warnings += 1

    print(f"\n{problems} conflitti, {warnings} avvisi")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
