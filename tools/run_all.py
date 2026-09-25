#!/usr/bin/env python3
"""Run every chapter's run.py (and optionally its figure script) and report.

Each script is executed from inside its own chapter directory, which is the
convention used throughout the book. Output is written to a log file per
chapter; the console shows one line per chapter with status and duration.

    python tools/run_all.py                    # all chapters, 20 min each
    python tools/run_all.py --timeout 120      # quick smoke test
    python tools/run_all.py --only 07 11 16    # selected chapters
    python tools/run_all.py --figures          # figure scripts instead
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = str(ROOT / ".venv" / "bin" / "python")
LOG_DIR = ROOT / "logs"

# Chapters needing a service or a large download; skipped unless asked for.
NEEDS_SERVICE = {"09": "Neo4j via docker-compose", "19": "large LLM download"}


def chapters(only):
    out = []
    for d in sorted(ROOT.glob("Chapter[0-9][0-9]")):
        num = d.name[-2:]
        if only and num not in only:
            continue
        out.append((num, d))
    return out


def scripts_for(chapter_dir, figures):
    """Every script to run for a chapter, as paths relative to it."""
    if figures:
        # Chapters 04 and 05 ship a grayscale and a heatmap variant beside the
        # main script; all of them produce figures used in the book.
        found = sorted(chapter_dir.glob("figures/generate_figures*.py"))
        return [str(p.relative_to(chapter_dir)) for p in found]
    return ["run.py"] if (chapter_dir / "run.py").exists() else []


def ssl_env():
    """Point urllib at certifi's CA bundle.

    The python.org build for macOS ships without CA certificates unless
    'Install Certificates.command' has been run, and chapters that download
    with urllib (14, 16) then fail with CERTIFICATE_VERIFY_FAILED.
    """
    if os.environ.get("SSL_CERT_FILE"):
        return {}
    try:
        import certifi
    except ImportError:
        return {}
    return {"SSL_CERT_FILE": certifi.where(), "REQUESTS_CA_BUNDLE": certifi.where()}


def run_one(num, chapter_dir, script, timeout):
    tag = "_" + Path(script).stem if "generate" in script else ""
    log_path = LOG_DIR / f"chapter{num}{tag}.log"
    started = time.time()
    with open(log_path, "w") as log:
        try:
            proc = subprocess.run(
                [PYTHON, script],
                cwd=chapter_dir,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                env={**os.environ, "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1",
                     **ssl_env()},
            )
            status = "OK" if proc.returncode == 0 else f"FAIL rc={proc.returncode}"
        except subprocess.TimeoutExpired:
            status = f"TIMEOUT >{timeout}s"
    return status, time.time() - started, log_path


def tail_error(log_path, lines=3):
    try:
        text = log_path.read_text(errors="ignore").rstrip().splitlines()
    except OSError:
        return ""
    return " | ".join(l.strip() for l in text[-lines:] if l.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--include-services", action="store_true",
                    help="also run chapters needing Neo4j or large downloads")
    args = ap.parse_args()

    LOG_DIR.mkdir(exist_ok=True)
    results = []
    for num, d in chapters(args.only):
        scripts = scripts_for(d, args.figures)
        if not scripts:
            results.append((num, "SKIP no script", 0.0, ""))
            print(f"Chapter{num}  SKIP   (nessuno script)", flush=True)
            continue
        if num in NEEDS_SERVICE and not args.include_services and not args.only:
            results.append((num, f"SKIP {NEEDS_SERVICE[num]}", 0.0, ""))
            print(f"Chapter{num}  SKIP   ({NEEDS_SERVICE[num]})", flush=True)
            continue

        for script in scripts:
            label = num if len(scripts) == 1 else f"{num} {Path(script).stem}"
            print(f"Chapter{label}  ...", end="", flush=True)
            status, secs, log_path = run_one(num, d, script, args.timeout)
            detail = tail_error(log_path) if status != "OK" else ""
            results.append((label, status, secs, detail))
            print(f"\rChapter{label}  {status:<14} {secs:6.1f}s  {script}", flush=True)
            if detail:
                print(f"            {detail[:160]}", flush=True)

    print("\n=== RIEPILOGO ===")
    width = max(len(s) for _, s, _, _ in results) if results else 4
    for num, status, secs, _ in results:
        print(f"Chapter{num}  {status:<{width}}  {secs:7.1f}s")
    failed = [n for n, s, _, _ in results if s.startswith(("FAIL", "TIMEOUT"))]
    print(f"\n{len(results)} capitoli, {len(failed)} con problemi"
          + (f": {', '.join(failed)}" if failed else ""))
    print(f"Log in {LOG_DIR}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
