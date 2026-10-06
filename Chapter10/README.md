# Chapter 10 – Graph Transformers: Attention Beyond Local Neighborhoods

Code for Chapter 10 of *Hands-On Graph Neural Networks Using Python*,
Second Edition (Packt).

## Files

| File | Purpose |
|---|---|
| `run.py` | Chapter code, run top to bottom (GINE baseline + GraphGPS) |
| `generate_figures.py` | Regenerates Figures 10.1–10.3 |
| `requirements.txt` | Python dependencies |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

No external service or GPU is required. ZINC-subset (12k molecules) is
downloaded automatically by PyG on first run. A GPU shortens training
time but is not necessary.

## Run

```bash
python run.py                # trains GINE and GraphGPS in sequence
python figures/generate_figures.py   # regenerates figures
```

Expected running time on CPU: ~5 minutes for Part 1 (GINE), ~15 minutes
for Part 2 (GraphGPS). On a modern GPU both parts complete in under a
minute.

## Expected results

Approximate test MAE on ZINC-subset with the hyperparameters in `run.py`:

- GINE baseline: ~0.35–0.40
- GraphGPS:      ~0.20–0.25

The reduction is roughly a factor of two. Actual numbers vary with seed
and PyTorch/PyG version.

## Updating Figure 10.3

`generate_figures.py` uses placeholder values for the stratified MAE bars
in Figure 10.3. After running `run.py` locally, replace the `gine` and
`gps` arrays at the top of the Figure 10.3 block with the real stratified
means printed by `run.py`.
