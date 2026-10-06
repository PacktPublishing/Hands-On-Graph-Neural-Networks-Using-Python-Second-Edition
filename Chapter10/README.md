# Chapter 10 – Graph Transformers: Attention Beyond Local Neighborhoods

A GINE baseline and a GraphGPS model on ZINC-subset, trained for 150 epochs
each, so the chapter can ask what global attention and Laplacian positional
encodings actually buy on molecular graphs.

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

`run.py` took about 53 minutes on the machine this was verified on, CPU only —
the two models are trained one after the other, 150 epochs each. A GPU is not
required but shortens this considerably.

## What to expect

Test MAE on ZINC-subset, from the verified run:

| model | test MAE |
|---|---|
| GINE baseline | 0.2400 |
| GraphGPS | 0.2091 |

That is a gain of about 13%, not the order-of-magnitude difference the
GraphGPS paper reports on other datasets — and the chapter treats that as the
interesting result rather than an embarrassment. ZINC molecules average 23
atoms, so six rounds of message passing already reach most of the graph, and
there is little left for global attention to add. Exact numbers vary with seed
and with the PyTorch/PyG version.

The stratified breakdown by molecule size, Figure 10.3, is where the two models
differ most: both degrade on the largest molecules, GINE more sharply.

## The ablation behind Figure 10.4

GINE and GraphGPS differ in two ways at once, positional encodings and global
attention, so the headline comparison cannot say which one matters.
[figures/ablation.ipynb](figures/ablation.ipynb) separates them with a 2x2
design at matched depth and width over several seeds, and produces
`fig10_4_ablation.png`. It is written for Colab, since it trains four variants.

## Figures

4 figures, listed in [figures/INDEX.md](figures/INDEX.md). Regenerate them with
the command above; they are drawn in grayscale, as printed.

The stratified bars in Figure 10.3 are hard-coded near the top of the Figure
10.3 block in `figures/generate_figures.py`, from an earlier run. After running
`run.py` yourself, replace the `gine` and `gps` arrays with the stratified means
it prints if you want the figure to match your own numbers exactly.

## Requirements

Covered by the pinned environment in the repository root. This chapter needs:

```
torch>=2.2
torch-geometric>=2.5
pandas>=2.0
matplotlib>=3.8
numpy>=1.26
```
