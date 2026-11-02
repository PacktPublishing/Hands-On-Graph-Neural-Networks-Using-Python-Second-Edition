# Chapter 14 – Temporal Graph Neural Networks

## Running it

```bash
cd Chapter14
python run.py
python figures/generate_figures.py
```

## Data

Downloaded on first run, into this folder:

- England COVID
- JODIE (Wikipedia interactions)
- WikiMaths

## Figures

15 figures, listed in [figures/INDEX.md](figures/INDEX.md). Regenerate them with the command above; they are drawn in grayscale, as printed.

## Requirements

Covered by the pinned environment in the repository root. This chapter needs:

```
torch==2.5.1
torch-geometric==2.8.0.post1
torch-scatter==2.1.2
torch-sparse==0.6.18
torch-geometric-temporal==0.56.2
pandas==2.3.3
matplotlib==3.10.8
scikit-learn==1.8.0
numpy==2.3.5
networkx==3.6.1
```
