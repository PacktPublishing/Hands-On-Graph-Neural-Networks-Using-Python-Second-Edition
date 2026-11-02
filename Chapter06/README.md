# Chapter 6 – Introducing Graph Convolutional Networks

## Running it

```bash
cd Chapter06
python run.py
python figures/generate_figures.py
```

`run.py` took about 2 minutes on the machine this was verified on, CPU only.

## Data

Downloaded on first run, into this folder:

- Cora (Planetoid)
- Facebook Page-Page
- Wikipedia Chameleon

## Figures

8 figures, listed in [figures/INDEX.md](figures/INDEX.md). Regenerate them with the command above; they are drawn in grayscale, as printed.

## Requirements

Covered by the pinned environment in the repository root. This chapter needs:

```
torch==2.5.1
torch-geometric==2.8.0.post1
scikit-learn==1.8.0
matplotlib==3.10.8
pandas==2.3.3
numpy==2.3.5
scipy==1.17.1
certifi==2026.5.20
```
