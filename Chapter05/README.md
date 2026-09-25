# Chapter 5 – Including Node Features with Vanilla Neural Networks

## Running it

```bash
cd Chapter05
python run.py
python figures/generate_figures.py
python figures/generate_figures_grayscale.py
```

`run.py` took about 2 minutes on the machine this was verified on, CPU only.

## Data

Downloaded on first run, into this folder:

- Cora / CiteSeer / PubMed (Planetoid)
- Facebook Page-Page

## Figures

7 figures, listed in [figures/INDEX.md](figures/INDEX.md). Regenerate them with the command above; they are drawn in grayscale, as printed.

## Requirements

Covered by the pinned environment in the repository root. This chapter needs:

```
torch>=2.2
torch-geometric>=2.5
scikit-learn>=1.4
pandas>=2.0
matplotlib>=3.8
numpy>=1.26
```
