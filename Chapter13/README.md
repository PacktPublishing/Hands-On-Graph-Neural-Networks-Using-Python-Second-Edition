# Chapter 13 – Learning from Heterogeneous Graphs

## Running it

```bash
cd Chapter13
python run.py
python figures/generate_figures.py
```

`run.py` took about 8 minutes on the machine this was verified on, CPU only. Almost all of it is HAN: the APCPA and APTPA meta-paths connect authors through 5.0 and 7.0 million edges.

## Data

Downloaded on first run, into this folder:

- DBLP

## Figures

6 figures, listed in [figures/INDEX.md](figures/INDEX.md), all drawn in grayscale at print width with `tools/bookstyle.py`. `figures/generate_figures.py` draws the five diagrams (13.1–13.5); `run.py` draws the comparison table (13.6) from the results of the run.

## Requirements

Covered by the pinned environment in the repository root. This chapter needs:

```
torch>=2.5.1
torch-geometric>=2.8
numpy>=2.3.5
matplotlib>=3.10.8
```
