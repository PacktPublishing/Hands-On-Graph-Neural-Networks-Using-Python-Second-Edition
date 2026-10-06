# Chapter 20 – Unlocking the Potential of Graph Foundation Models

The closing chapter is about ideas and frameworks rather than code: where graph
foundation models stand, what pretraining them means, and what is still
unsolved. There is no runnable demonstration, and no `run.py` — the only code
here draws the two figures.

## What is in the chapter

1. **From Specialized GNNs to Foundation Models** – motivation, the parallel
   with the transitions in NLP and computer vision, and why graphs are harder.
2. **Pretraining Objectives for Graphs** – five families of pretraining
   strategy: contrastive, masked modeling, LLM-augmented, multi-graph,
   multi-task.
3. **Graph Foundation Models in the Wild** – three case studies:
   recommendation at web scale, drug discovery (MolFM, GEM-2, Uni-Mol), and
   knowledge graph completion (ULTRA, GFM-RAG).
4. **Challenges of Graph Foundation Models** – scaling laws, cross-domain
   transfer, evaluation and contamination, and what "general" should even mean
   for a graph model.
5. **Where to Go from Here** – directions for readers who want to deepen the
   theory, deploy in production, work with LLMs, or contribute to the field.

## Figures

2 figures, listed in [figures/INDEX.md](figures/INDEX.md):

- **Figure 20.1** – the specialized workflow (one model per graph per task)
  against the foundation workflow (one pretrained backbone reused across
  tasks).
- **Figure 20.2** – the five families of pretraining approach compared.

Regenerate them with:

```bash
cd Chapter20
python figures/generate_figures.py
```

They are drawn in grayscale, as printed. `matplotlib` is the only dependency.
