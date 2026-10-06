# Chapter 19 – Large Language Models Meet Graph Neural Networks

Code for Chapter 19 of *Hands-On Graph Neural Networks Using Python*,
Second Edition (Packt).

## Files

| File | Purpose |
|---|---|
| `run.py` | G-Retriever training + text-only baseline + comparison |
| `generate_figures.py` | Regenerates Figures 19.1, 19.2, 19.3 |
| `requirements.txt` | Python dependencies |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Approximately 2 GB of disk are needed for the Qwen3-0.6B model and the
WebQSP-tiny dataset (both downloaded on first run).

`pcst_fast` is a C++ extension that compiles at install time. On macOS
with Apple Silicon this requires the Xcode command-line tools
(`xcode-select --install`). On Linux, `build-essential` is enough.

## Run

```bash
python run.py                # trains G-Retriever, then runs comparison
python figures/generate_figures.py   # regenerates figures
```

Expected running time on CPU (Apple Silicon): ~30 minutes for training,
plus a few seconds per inference. On GPU, the same run completes in
minutes.

## What to expect

The chapter is deliberately configured for reproducibility on a laptop:

- LLM = Qwen3-0.6B (600 M parameters, CPU-friendly)
- Dataset = WebQSP-tiny (~500 examples)
- Training = 3 epochs, small learning rate

For this reason **no Hit@1 numbers are reported**. The comparison in
Part 3 is qualitative: for each of a few question types (single-hop,
multi-hop, aggregation), the script prints the expected answer,
G-Retriever's answer, and the text-only baseline's answer.

Expect small-LLM artifacts: incomplete answers, occasional
hallucination, repeated tokens. The point is to observe whether the
*type of error* differs between the two pipelines, not to measure
absolute quality.

## Caveats

- Question indices in Part 3 (`categories = {'single_hop': 0, ...}`) are
  placeholders. WebQSP-tiny does not label questions by reasoning type;
  inspect a few examples and replace the indices with genuine
  representatives before drawing conclusions.
- `torch_geometric.llm.models.GRetriever` is available from PyG 2.7.
  Earlier versions ship the class as `torch_geometric.nn.models.GRetriever`
  with a slightly different constructor (see the PyG changelog).
- Qwen3-0.6B requires no HuggingFace access approval. If you switch to
  Llama-3, request access via the model card first.
