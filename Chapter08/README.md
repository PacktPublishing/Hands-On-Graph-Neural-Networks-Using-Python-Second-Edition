# Chapter 8 – Scaling Up Graph Neural Networks with GraphSAGE

## Running it

```bash
cd Chapter08
python run.py
python figures/generate_figures.py
```

## Data

Downloaded on first run, into this folder:

- Cora / CiteSeer / PubMed (Planetoid)
- PPI

## Figures

6 figures, listed in [figures/INDEX.md](figures/INDEX.md). Regenerate them with the command above; they are drawn in grayscale, as printed.

## Requirements

Covered by the pinned environment in the repository root. This chapter needs:

```
torch>=2.2
torch-geometric>=2.5
pyg-lib
scikit-learn>=1.4
networkx>=3.4
matplotlib>=3.8
numpy>=1.26
```
