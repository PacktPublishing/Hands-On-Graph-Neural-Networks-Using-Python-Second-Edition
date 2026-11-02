# Chapter 8 – Scaling Up Graph Neural Networks with GraphSAGE

## Running it

```bash
cd Chapter08
python run.py
python figures/generate_figures.py
```

`run.py` did not finish within 90 minutes on the machine this was verified on, CPU only. Expect to leave it running, or to reduce the number of epochs.

## Data

Downloaded on first run, into this folder:

- Cora / CiteSeer / PubMed (Planetoid)
- PPI

## Figures

6 figures, listed in [figures/INDEX.md](figures/INDEX.md). Regenerate them with the command above; they are drawn in grayscale, as printed.

## Requirements

Covered by the pinned environment in the repository root. This chapter needs:

```
torch>=2.5.1
torch-geometric>=2.8
pyg-lib>=0.4
scikit-learn>=1.8
networkx>=3.6.1
matplotlib>=3.10.8
numpy>=2.3.5
```
