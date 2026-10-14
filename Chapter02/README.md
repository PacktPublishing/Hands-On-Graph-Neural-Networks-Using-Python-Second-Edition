# Chapter 2 – Graph Theory for Graph Neural Networks

The vocabulary the rest of the book relies on: what a graph is, how it is
represented, and the properties that later chapters assume you have met. The
chapter is pen-and-paper, so there is no `run.py` — the only code here draws
the figure.

## Figures

3 figures, listed in [figures/INDEX.md](figures/INDEX.md): a disconnected
graph beside a connected one, the common types of graph, and a graph beside
its adjacency matrix.

Figures 2.1, 2.2, 2.3, 2.7 and 2.8 are not here. They were accepted as
supplied and have no source in this repository, so the chapter's printed
figures come from two places.

Regenerate them with:

```bash
cd Chapter02
python figures/generate_figures.py
```

They are drawn in grayscale, as printed. `matplotlib` and `numpy` are the only dependencies.
