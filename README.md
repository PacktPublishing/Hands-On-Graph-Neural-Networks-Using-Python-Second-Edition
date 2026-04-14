# Hands-On Graph Neural Networks Using Python — Second Edition

Companion code for *Hands-On Graph Neural Networks Using Python, Second Edition*, published by Packt.

This repository contains per-chapter runnable scripts, figure generators, and pinned requirements so you can reproduce the examples from the book.

## Table of Contents

- [Chapter03/](Chapter03/) — Creating Node Representations with DeepWalk
- [Chapter04/](Chapter04/) — Improving Embeddings with Biased Random Walks in Node2Vec
- [Chapter05/](Chapter05/) — Including Node Features with Vanilla Neural Networks

Each chapter folder contains:
- `run.py` — the main script for the chapter
- `requirements.txt` — pinned Python dependencies
- `figures/` — scripts that regenerate the figures used in the book

## Setup

The code targets Python 3.10+.

1. Clone the repository:
   ```bash
   git clone https://github.com/PacktPublishing/Hands-On-Graph-Neural-Networks-Using-Python-Second-Edition.git
   cd Hands-On-Graph-Neural-Networks-Using-Python-Second-Edition
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # on Windows: .venv\Scripts\activate
   ```

3. Install the dependencies for the chapter you want to run:
   ```bash
   pip install -r Chapter03/requirements.txt
   ```

## Running a chapter

From the repository root:

```bash
python Chapter03/run.py
```

To regenerate the figures for a chapter:

```bash
python Chapter03/figures/generate_figures.py
```

## License

See [LICENSE](LICENSE).
