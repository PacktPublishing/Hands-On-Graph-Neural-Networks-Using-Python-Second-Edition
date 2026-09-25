#!/usr/bin/env python3
"""Write a README.md for each chapter from what is actually in the chapter.

Everything in the generated file is read from the chapter itself — the title
and summary from run.py's docstring, the datasets from the dataset classes the
code instantiates, the figure count from figures/, the dependencies from
requirements.txt — so nothing here is a description someone has to keep true by
hand.

Measured run times, if tools/run_all.py has been run, are read from its
summary log. Chapters that already have a hand-written README are left alone
unless --force is given.

    python tools/make_chapter_readme.py
    python tools/make_chapter_readme.py --only 11 13
    python tools/make_chapter_readme.py --force     # overwrite existing ones
"""

import argparse
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "logs" / "run_all_summary.txt"

# Dataset class -> what the reader ends up downloading.
DATASETS = {
    "Planetoid": "Cora / CiteSeer / PubMed (Planetoid)",
    "TUDataset": "TUDataset (PROTEINS, MUTAG)",
    "PPI": "PPI",
    "DBLP": "DBLP",
    "ZINC": "ZINC",
    "Amazon": "Amazon Photo",
    "FacebookPagePageSNAP": "Facebook Page-Page",
    "WikipediaChameleonSNAP": "Wikipedia Chameleon",
    "PygNodePropPredDataset": "OGB (ogbn-arxiv)",
    "WikiMathsDatasetLoader": "WikiMaths",
    "EnglandCovidDatasetLoader": "England COVID",
    "JODIEDataset": "JODIE (Wikipedia interactions)",
    "WebQSPDataset": "WebQSP",
}


def runtimes():
    if not SUMMARY.exists():
        return {}
    out = {}
    for line in SUMMARY.read_text(errors="ignore").splitlines():
        m = re.match(r"Chapter(\d\d)\s+(OK|FAIL[^\s]*|TIMEOUT[^\s]*|SKIP.*?)\s+([\d.]+)s\s*$",
                     line.strip())
        if m:
            out[m.group(1)] = (m.group(2), float(m.group(3)))
    return out


def summarise(run_py):
    """(title, the paragraph under it) from the module docstring."""
    doc = ast.get_docstring(ast.parse(run_py.read_text())) or ""
    lines = [l.rstrip() for l in doc.strip().splitlines()]
    if not lines:
        return None, []
    title = lines[0]
    body = []
    for line in lines[1:]:
        if re.match(r"^\s*(Requirements|Key fixes|Figure generation)", line):
            break
        # every docstring repeats the book title under the chapter title
        if "Hands-On Graph Neural Networks" in line:
            continue
        body.append(line)
    return title, [l for l in body if l.strip()]


def build(chapter_dir, times):
    num = chapter_dir.name[-2:]
    run_py = chapter_dir / "run.py"
    title, body = summarise(run_py) if run_py.exists() else (None, [])
    lines = [f"# {title or chapter_dir.name}", ""]
    lines += body + [""] if body else []

    lines += ["## Running it", "", "```bash", f"cd {chapter_dir.name}"]
    if run_py.exists():
        lines.append("python run.py")
    for script in sorted(chapter_dir.glob("figures/generate_figures*.py")):
        lines.append(f"python figures/{script.name}")
    lines += ["```", ""]

    status = times.get(num)
    if status and status[0] == "OK":
        mins = status[1] / 60
        took = f"{status[1]:.0f} seconds" if status[1] < 90 else f"about {mins:.0f} minutes"
        lines += [f"`run.py` took {took} on the machine this was verified on, "
                  "CPU only.", ""]
    elif status and status[0].startswith("TIMEOUT"):
        lines += ["`run.py` trains for a long time — over 15 minutes on the "
                  "machine this was verified on, CPU only.", ""]

    if run_py.exists():
        used = sorted({DATASETS[k] for k in DATASETS if k in run_py.read_text()})
        if used:
            lines += ["## Data", "",
                      "Downloaded on first run, into this folder:", ""]
            lines += [f"- {d}" for d in used] + [""]

    figs = sorted((chapter_dir / "figures").glob("*.png")) if (chapter_dir / "figures").is_dir() else []
    if figs:
        lines += ["## Figures", "",
                  f"{len(figs)} figures, listed in "
                  f"[figures/INDEX.md](figures/INDEX.md). Regenerate them with "
                  f"the command above; they are drawn in grayscale, as printed.",
                  ""]

    req = chapter_dir / "requirements.txt"
    if req.exists():
        pkgs = [l.split("#")[0].strip() for l in req.read_text().splitlines()]
        pkgs = [p for p in pkgs if p and not p.startswith("-")]
        lines += ["## Requirements", "",
                  "Covered by the pinned environment in the repository root. "
                  "This chapter needs:", "",
                  "```", *pkgs, "```", ""]
    return "\n".join(lines).rstrip() + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    times = runtimes()
    for chapter_dir in sorted(ROOT.glob("Chapter[0-9][0-9]")):
        num = chapter_dir.name[-2:]
        if args.only and num not in args.only:
            continue
        readme = chapter_dir / "README.md"
        if readme.exists() and not args.force:
            print(f"{chapter_dir.name}: README già presente, lasciato com'è")
            continue
        if not (chapter_dir / "run.py").exists():
            print(f"{chapter_dir.name}: nessun run.py, saltato")
            continue
        readme.write_text(build(chapter_dir, times))
        print(f"{readme.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
