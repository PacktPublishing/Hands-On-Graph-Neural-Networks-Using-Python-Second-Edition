# Chapter 19 – Large Language Models Meet Graph Neural Networks

G-Retriever on ExplaGraphs: a GNN encodes the graph into a soft prompt, the
same graph also goes into the prompt as text, and a small LLM answers. The
chapter evaluates that against baselines that drop one ingredient at a time.

## Setup

```bash
cd Chapter19
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The LLM is Qwen3-0.6B, small enough to run on a laptop CPU and needing no
HuggingFace access approval. ExplaGraphs is downloaded from GitHub on first
run, into `data/`.

## Trained weights

`run.py` and `ablation.py` do not train. They read the weights from
`checkpoints/` and, when a file is missing, download it from Hugging Face
(`giuseppefutia/hands-on-gnn-ch19-gretriever`) automatically. Nothing to do by
hand.

The weights are not in git — `epoch_10.pt` alone is 2.4 GB. The small `.json`
results they produce are, so the figures can be regenerated without a GPU.

To reproduce the weights instead of downloading them, run
[train_colab.ipynb](train_colab.ipynb) and
[ablation_colab.ipynb](ablation_colab.ipynb) on a GPU, then
`export_checkpoint.py` to publish them.

## Running it

```bash
python run.py                        # the three systems of Figure 19.3
python ablation.py                   # adds the two of Figure 19.4
python figures/generate_figures.py   # conceptual figures 19.1 and 19.2
```

`ablation.py` reads `checkpoints/comparison.json`, so run `run.py` first.
Inference over the 398 test samples takes a few minutes on CPU.

## What to expect

Accuracy on the 398 ExplaGraphs test samples, from the verified run:

| system | graph in the prompt | GNN soft prompt | accuracy |
|---|---|---|---|
| G-Retriever | yes | yes | 86.7% |
| G-Retriever, soft prompt removed at inference | yes | no | 87.7% |
| LoRA baseline, no GNN | yes | no | 87.2% |
| G-Retriever trained on the soft prompt only | no | yes | 80.4% |
| LoRA trained without any graph | no | no | 81.2% |

Read the table by column rather than by row. The graph is worth about six
points, and it is the **linearized graph in the prompt** that carries them: the
three systems that see it land within a point of each other, and the two that
do not fall together. The GNN soft prompt contributes nothing measurable here —
removing it at inference even scores marginally higher, which is within noise.

This is a negative result and the chapter keeps it. ExplaGraphs graphs are tiny,
a handful of triples, so a language model reads them perfectly well as text and
has little use for a learned graph embedding. The soft prompt is the mechanism
worth understanding; this dataset is not where it pays off.

## Caveats

- `torch_geometric.llm.models.GRetriever` is used here, which is where the
  class lives from PyG 2.7 on. Earlier versions ship it as
  `torch_geometric.nn.models.GRetriever` with a slightly different constructor
  (see the PyG changelog). `requirements.txt` pins 2.8.
- If you switch to a gated model such as Llama-3, request access on its model
  card first.
- Expect small-LLM artifacts in the per-sample output: incomplete answers,
  occasional repetition. The accuracy above is computed on the normalized
  answer, not the raw generation.

## Figures

4 figures, listed in [figures/INDEX.md](figures/INDEX.md). Figures 19.1 and
19.2 are conceptual and come from `figures/generate_figures.py`; 19.3 is
written by `run.py` and 19.4 by `ablation.py`.

## Requirements

Covered by the pinned environment in the repository root. This chapter needs:

```
torch>=2.2
torch_geometric==2.8.0
transformers>=4.44
accelerate>=0.30
peft>=0.12
sentencepiece>=0.2
huggingface_hub>=0.23
numpy>=1.26
matplotlib>=3.8
```
