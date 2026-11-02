# Chapter 7 – Graph Attention Networks

## Running it

```bash
cd Chapter07
python run.py
python figures/generate_figures.py
```

`run.py` took about 3 minutes on the machine this was verified on, CPU only.

## Data

Downloaded on first run, into this folder:

- Cora / CiteSeer / PubMed (Planetoid)

## Figures

6 figures, listed in [figures/INDEX.md](figures/INDEX.md). Regenerate them with the command above; they are drawn in grayscale, as printed.

## Requirements

Covered by the pinned environment in the repository root. This chapter needs:

```
torch>=2.5.1
torch-geometric>=2.8
matplotlib>=3.10.8
numpy>=2.3.5
networkx>=3.6.1
```
