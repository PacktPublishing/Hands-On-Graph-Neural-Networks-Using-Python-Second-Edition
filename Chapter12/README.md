# Chapter 12 – Predicting Links with Graph Neural Networks

## Running it

```bash
cd Chapter12
python run.py
python figures/generate_figures.py
```

`run.py` took 73 seconds on the machine this was verified on, CPU only.

## Data

Downloaded on first run, into this folder:

- Cora / CiteSeer / PubMed (Planetoid)

## Figures

4 figures, listed in [figures/INDEX.md](figures/INDEX.md). Regenerate them with the command above; they are drawn in grayscale, as printed.

## Requirements

Covered by the pinned environment in the repository root. This chapter needs:

```
torch==2.5.1
torch-geometric==2.8.0.post1
scikit-learn==1.8.0
scipy==1.17.1
numpy==2.3.5
matplotlib==3.10.8
networkx==3.6.1
```
