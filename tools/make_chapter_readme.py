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
# The quick campaign, then the long one that re-runs whatever it could not
# finish. Later wins: a chapter that timed out at 900s and then completed in
# 3164s is an OK chapter that takes 53 minutes, not a timeout.
SUMMARIES = [ROOT / "logs" / "run_all_summary.txt",
             ROOT / "logs" / "run_slow_summary.txt"]

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
    """{chapter: (status, seconds)} from the run campaigns.

    The status field is free text after the verdict — "TIMEOUT >900s",
    "FAIL rc=-9", "SKIP Neo4j via docker-compose" — so the seconds are taken
    from the end of the line and the verdict from its first word. An earlier
    version anchored the seconds straight after the verdict, which silently
    dropped every FAIL and TIMEOUT line: the chapters a reader most needs
    warning about were the ones with no measured time at all.
    """
    out = {}
    for summary in SUMMARIES:
        if not summary.exists():
            continue
        for line in summary.read_text(errors="ignore").splitlines():
            m = re.match(r"Chapter(\d\d)\s+(OK|FAIL|TIMEOUT|SKIP)\b.*?([\d.]+)s\s*$",
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
        # The summary is the prose under the title. Everything from the first
        # structural heading on is for someone reading run.py — the per-part
        # walkthrough, the requirements, the list of files it writes — and
        # belongs in the notebook, not in the README.
        if re.match(r"^\s*(Requirements|Key fixes|Figure generation"
                    r"|PARTE?\s*\d|Parts?\s*\d|Figures produced)", line):
            break
        # every docstring repeats the book title under the chapter title
        if "Hands-On Graph Neural Networks" in line:
            continue
        body.append(line)
    return title, [l for l in body if l.strip()]


# ---------------------------------------------------------------------------
# Keeping an existing README in step, without rewriting what someone wrote.
#
# Several chapters have a hand-written README: Neo4j setup, expected results,
# caveats. Regenerating those would throw that away. What *does* go stale is
# the handful of blocks read from files — the requirements list and the figure
# count. sync() refreshes exactly those, and only when the README already has
# the generated block, so a hand-written one is never given sections it never
# had.
# ---------------------------------------------------------------------------

# The measured-runtime sentence, in one place, so build() and sync() cannot
# word it differently.
# Only the generated sentence, whole and on its own line. A hand-written
# README may open the same way and then carry on ("... CPU only — the two
# models are trained one after the other"); replacing that would behead the
# author's sentence to restore a number that already agreed.
RUNTIME_LINE = re.compile(
    r"^`run\.py` (?:took [^\n]*?CPU only\.|"
    r"did not finish within [^\n]*?epochs\.|"
    r"trains for a long time[^\n]*?\.)$",
    re.MULTILINE)


def runtime_sentence(status):
    if not status:
        return None
    verdict, secs = status
    if verdict == "OK":
        took = f"{secs:.0f} seconds" if secs < 90 else f"about {secs / 60:.0f} minutes"
        return (f"`run.py` took {took} on the machine this was verified on, "
                "CPU only.")
    if verdict == "TIMEOUT":
        return (f"`run.py` did not finish within {secs / 60:.0f} minutes on the "
                "machine this was verified on, CPU only. Expect to leave it "
                "running, or to reduce the number of epochs.")
    return None


REQ_BLOCK = re.compile(
    r"(## Requirements\n\nCovered by the pinned environment[^\n]*\n\n```\n)"
    r"(.*?)"
    r"(```)",
    re.DOTALL)
FIG_COUNT = re.compile(r"^\d+ figures, listed in \[figures/INDEX\.md\]", re.MULTILINE)


def packages(chapter_dir):
    """The dependency lines of requirements.txt, as the README lists them."""
    req = chapter_dir / "requirements.txt"
    if not req.exists():
        return None
    pkgs = [l.split("#")[0].strip() for l in req.read_text().splitlines()]
    return [p for p in pkgs if p and not p.startswith("-")]


def sync(text, chapter_dir, times=None):
    """Refresh the derived blocks of an existing README. Returns the new text."""
    pkgs = packages(chapter_dir)
    if pkgs is not None:
        text = REQ_BLOCK.sub(
            lambda m: m.group(1) + "\n".join(pkgs) + "\n" + m.group(3), text)

    sentence = runtime_sentence((times or {}).get(chapter_dir.name[-2:]))
    if sentence and RUNTIME_LINE.search(text):
        text = RUNTIME_LINE.sub(lambda _: sentence, text, count=1)

    figs_dir = chapter_dir / "figures"
    figs = sorted(figs_dir.glob("*.png")) if figs_dir.is_dir() else []
    if figs:
        text = FIG_COUNT.sub(
            f"{len(figs)} figures, listed in [figures/INDEX.md]", text)
    return text


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

    # FAIL is deliberately not reported: the summary is only as fresh as the
    # last campaign, and a chapter fixed since then would be libelled by it.
    sentence = runtime_sentence(times.get(num))
    if sentence:
        lines += [sentence, ""]

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
    ap.add_argument("--force", action="store_true",
                    help="rewrite READMEs from scratch, hand-written ones included")
    ap.add_argument("--sync", action="store_true",
                    help="refresh only the derived blocks of existing READMEs")
    ap.add_argument("--check", action="store_true",
                    help="report drift instead of writing")
    args = ap.parse_args()

    times = runtimes()
    stale = []
    for chapter_dir in sorted(ROOT.glob("Chapter[0-9][0-9]")):
        num = chapter_dir.name[-2:]
        if args.only and num not in args.only:
            continue
        readme = chapter_dir / "README.md"

        if not (chapter_dir / "run.py").exists():
            if not (args.check or args.sync):
                print(f"{chapter_dir.name}: nessun run.py, saltato")
            continue

        if not readme.exists():
            if args.check:
                stale.append((readme, "manca"))
            else:
                readme.write_text(build(chapter_dir, times))
                print(f"{readme.relative_to(ROOT)}  (creato)")
            continue

        if args.sync or args.check:
            current = readme.read_text()
            updated = sync(current, chapter_dir, times)
            if updated == current:
                continue
            if args.check:
                stale.append((readme, "blocchi derivati non aggiornati"))
            else:
                readme.write_text(updated)
                print(f"{readme.relative_to(ROOT)}  (blocchi derivati aggiornati)")
            continue

        if not args.force:
            print(f"{chapter_dir.name}: README già presente, lasciato com'è")
            continue

        readme.write_text(build(chapter_dir, times))
        print(f"{readme.relative_to(ROOT)}")

    if args.check:
        for path, why in stale:
            print(f"STALE  {path.relative_to(ROOT)} — {why}")
        print(f"\n{len(stale)} README non allineati")
        return 1 if stale else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
