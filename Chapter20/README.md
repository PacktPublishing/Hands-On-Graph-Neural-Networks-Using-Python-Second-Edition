# Chapter 20 – Unlocking the Potential of Graph Foundation Models
## Theoretical variant (no runnable demo)

Code for Chapter 20 of *Hands-On Graph Neural Networks Using Python*,
Second Edition (Packt).

## Which variant is this?

This is the **theoretical** variant of Chapter 20, aligned with the
original book outline which classifies this chapter as *"more about ideas
and frameworks than hands-on coding"*. It contains no runnable
demonstration.

A separate variant with a runnable GraphAny demonstration exists in the
`Chapter20/` folder (not this one). The two variants share Sections 1, 2,
and 4, and diverge in Section 3 and in the closing pages.

## Files

| File | Purpose |
|---|---|
| `Chapter20_..._THEORETICAL.docx` | The chapter manuscript |
| `generate_figures.py` | Regenerates Figure 20.1 |
| `fig20_1_specialized_vs_foundation.png` | Conceptual figure |

No `run.py`, no `requirements.txt` beyond matplotlib for the figure
regeneration. The chapter is deliberately code-free.

## What is in the chapter

Five sections:

1. **From Specialized GNNs to Foundation Models** – motivation, parallel
   with NLP and computer vision transitions, why graphs are harder.
2. **Pretraining Objectives for Graphs** – five families of pretraining
   strategies: contrastive, masked modeling, LLM-augmented, multi-graph,
   multi-task.
3. **Graph Foundation Models in the Wild: Three Case Studies** –
   recommendation at web scale (Snap, Spotify style backbones), drug
   discovery (MolFM, GEM-2, Uni-Mol), knowledge graph completion
   (ULTRA, GFM-RAG).
4. **Challenges of Graph Foundation Models** – scaling laws, cross-domain
   transfer, evaluation/contamination, definition of generality.
5. **Where to Go from Here** – map of directions for readers who want to
   deepen theory, deploy in production, work with LLMs, or contribute to
   the field.

## Figures

- **Figure 20.1**: side-by-side conceptual diagram of the specialized
  workflow (one model per graph per task) vs the foundation workflow
  (one pretrained backbone reused across tasks).

To regenerate: `python generate_figures.py`.

## Notes

- Length: ~14–15 pages (slightly longer than the runnable-demo variant
  because Section 3 was expanded from a demo to three case studies, and
  a new closing section was added).
- No external dependencies at read time. The chapter can be reviewed
  by editorial staff without any Python environment.
