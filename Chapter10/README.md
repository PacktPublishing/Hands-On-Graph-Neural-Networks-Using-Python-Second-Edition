# Chapter 10 – Graph Transformers: Attention Beyond Local Neighborhoods

A GINE baseline and a GraphGPS model on ZINC-subset, trained for 150 epochs
each, followed by an ablation that asks what global attention, Laplacian
positional encodings, and the rest of the GraphGPS block actually contribute
on molecular graphs.

## Setup

```bash
cd Chapter10
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

No external service is required. ZINC-subset (12k molecules) is downloaded by
PyG on first run.

## Running it

```bash
python run.py                        # trains GINE, then GraphGPS
python figures/generate_figures.py   # regenerates figures
```

`run.py` took about 34 minutes on the machine this was verified on, CPU only:
about 2.7 seconds per epoch for GINE and 10.8 for GraphGPS, 150 epochs each.
A GPU is not required but shortens this considerably.

## What to expect

Test MAE on ZINC-subset, from the verified run:

| model | test MAE |
|---|---|
| GINE baseline | 0.2416 |
| GraphGPS | 0.2025 |

That is a reduction of about 16%. Exact numbers vary with seed and with the
PyTorch/PyG version; the run above was produced with PyTorch 2.5.1,
PyTorch Geometric 2.8.0, and pandas 2.3.3.

The stratified breakdown by molecule size, Figure 10.3, shows that the error is
not monotonic in molecule size: both models do best on the middle bins,
21-25 atoms, and worse at both ends.

## The ablation behind Figure 10.4

GINE and GraphGPS differ in several ways at once: global attention, positional
encodings, and the rest of the GraphGPS block (residual connections,
normalization, a feed-forward layer, dropout, more layers and channels).
[ablation.ipynb](ablation.ipynb) separates them with six
variants over three seeds, and its results feed `fig10_4_ablation.png`:

| variant | parameters | test MAE (3 seeds) |
|---|---|---|
| GINE baseline (4 layers, 64 hidden) | 40,069 | 0.2551 ± 0.0286 |
| local only, no PE | 349,447 | 0.1920 ± 0.0077 |
| local only + PE | 350,311 | 0.1898 ± 0.0023 |
| local only, widened | 570,604 | 0.1795 ± 0.0011 |
| attention, no PE | 574,087 | 0.2210 ± 0.0062 |
| attention + PE (full GraphGPS) | 574,951 | 0.1941 ± 0.0021 |

The first row is the same GINE baseline as above, re-run over three seeds on
different hardware, which is why it reads 0.2551 rather than 0.2416; the single
run falls well inside the spread. The "local only" variants are the GraphGPS
block with its attention branch removed. On ZINC-subset, the gain over GINE
comes from the block design rather than from global attention, consistently
with the ablation in the GraphGPS paper. The notebook is written for Colab and
was run on an A100 GPU with PyTorch 2.11 and PyTorch Geometric 2.8.1; the full
grid takes about one hour and forty minutes there.

## Figures

4 figures, listed in [figures/INDEX.md](figures/INDEX.md). Regenerate them with
the command above; they are drawn in grayscale, as printed.

The stratified bars in Figure 10.3 and the ablation rows in Figure 10.4 are
hard-coded in `figures/generate_figures.py`. They match the verified runs to
four decimals; if you re-run the chapter or the ablation, replace them with the
values printed by `run.py` and `ablation.ipynb`.

## Requirements

Covered by the pinned environment in the repository root. This chapter needs:

```
torch>=2.2
torch-geometric>=2.5
pandas>=2.0
matplotlib>=3.8
numpy>=1.26
```
