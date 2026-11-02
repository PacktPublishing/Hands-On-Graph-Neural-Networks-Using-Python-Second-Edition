# Chapter 15 – Explaining Graph Neural Networks

## Running it

```bash
cd Chapter15
python run.py
python figures/generate_figures.py
```

`run.py` took 40 seconds on the machine this was verified on, CPU only.

## Data

Downloaded on first run, into this folder:

- Amazon Photo
- TUDataset (PROTEINS, MUTAG)

## Figures

5 figures, listed in [figures/INDEX.md](figures/INDEX.md). Regenerate them with the command above; they are drawn in grayscale, as printed.

## Requirements

Covered by the pinned environment in the repository root. This chapter needs:

```
torch>=2.5.1
torch-geometric>=2.8
captum>=0.9
numpy>=2.3.5
scipy>=1.17.1
networkx>=3.6.1
matplotlib>=3.10.8
```
