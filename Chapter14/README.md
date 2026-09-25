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
torch>=2.2
torch-geometric>=2.5
torch-geometric-temporal==0.54.0
pandas>=2.0
matplotlib>=3.8
scikit-learn>=1.4
numpy>=1.26
```
