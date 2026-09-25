"""
Chapter 19 – Large Language Models Meet Graph Neural Networks
Hands-On Graph Neural Networks Using Python (2nd Edition)

Requirements:
    pip install torch torch_geometric>=2.7 transformers datasets \
                sentencepiece accelerate pcst_fast rank-bm25

Prerequisites:
    - Approximately 2 GB of disk for the Qwen3-0.6B model and the
      WebQSP-tiny dataset (both downloaded on first run).
    - Xcode command-line tools on macOS (`xcode-select --install`)
      for compiling pcst_fast; build-essential on Linux.
    - No GPU required. Training on CPU (Apple Silicon) takes ~30 minutes.

New in this chapter (no first-edition counterpart):
  - Part 1: G-Retriever end-to-end training on WebQSP-tiny using
            torch_geometric.llm.models.GRetriever.
  - Part 2: text-only RAG baseline with BM25 retrieval on a linearized
            graph, using the same LLM (Qwen3-0.6B) without any GNN.
  - Part 3: qualitative side-by-side comparison on selected questions.

Design notes:
  - The LLM is Qwen3-0.6B for CPU reproducibility. Llama-3.1-8B would
    perform better but requires HuggingFace access approval and a GPU.
  - The training run is deliberately short (3 epochs) to demonstrate the
    pipeline without a compute budget beyond a laptop. Production runs
    use more epochs and a larger LLM.
  - We do NOT report Hit@1 numbers. With a 0.6B LLM and a 500-example
    subset, no ranking would be robust enough to draw a conclusion.
    The comparison in Part 3 is qualitative by design.
"""

import torch
import numpy as np
import time
from contextlib import contextmanager
from typing import List

SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ── Timing instrumentation ──────────────────────────────────────────────────
#
# Each phase of the pipeline is wrapped in `with timed(name):` so we can
# report a compact summary at the end. Useful for deciding what to cache
# between runs and what to precompute for readers of the book.

timings: dict = {}

@contextmanager
def timed(name: str):
    start = time.time()
    try:
        yield
    finally:
        timings[name] = time.time() - start

def print_timings() -> None:
    print("\n" + "=" * 60)
    print("Timing summary")
    print("=" * 60)
    width = max(len(n) for n in timings) if timings else 0
    for name, seconds in timings.items():
        print(f"  {name:<{width}} : {seconds:>7.1f}s "
              f"({seconds/60:>5.2f} min)")
    total = sum(timings.values())
    print("-" * 60)
    print(f"  {'TOTAL':<{width}} : {total:>7.1f}s "
          f"({total/60:>5.2f} min)")


# =============================================================================
# PART 1 – Load Qwen3-0.6B, WebQSP-tiny, and train G-Retriever
# =============================================================================

print("\n" + "=" * 60)
print("PART 1 – G-Retriever training on WebQSP-tiny")
print("=" * 60)

from torch_geometric.llm.models import LLM, GRetriever
from torch_geometric.nn         import GAT
from torch_geometric.datasets   import WebQSPDataset
from torch_geometric.loader     import DataLoader

# --- Load the LLM (downloads ~1.5 GB on first run) --------------------------
print("\nLoading Qwen3-0.6B …")
with timed("load_qwen3"):
    llm = LLM(
        model_name='Qwen/Qwen3-0.6B',
        num_params=0.6,
        dtype=torch.bfloat16,
    )

# --- Load the dataset -------------------------------------------------------
print("\nLoading WebQSP-tiny …")
with timed("webqsp_load_and_index"):
    train_ds = WebQSPDataset(root='./data/WebQSP', split='train')
    val_ds   = WebQSPDataset(root='./data/WebQSP', split='val')
    test_ds  = WebQSPDataset(root='./data/WebQSP', split='test')
print(f"  train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
print(f"  sample: {train_ds[0]}")

# --- Assemble the model -----------------------------------------------------
print("\nAssembling G-Retriever (GAT encoder + Qwen3-0.6B + LoRA) …")
with timed("assemble_gretriever"):
    gnn = GAT(
        in_channels=1024,
        hidden_channels=1024,
        num_layers=4,
        out_channels=1024,
        heads=4,
    )

    model = GRetriever(
        llm=llm,
        gnn=gnn,
        use_lora=True,
        mlp_out_tokens=1,
    )

# --- Training loop ----------------------------------------------------------
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-5,
)

print("\nTraining G-Retriever (3 epochs) …")
with timed("training_total"):
    for epoch in range(1, 4):
        with timed(f"training_epoch_{epoch}"):
            model.train()
            total_loss = 0.0
            n_seen = 0
            for batch in train_loader:
                optimizer.zero_grad()
                loss = model(
                    question=batch.question,
                    x=batch.x.float(),
                    edge_index=batch.edge_index,
                    batch=batch.batch,
                    label=batch.label,
                    edge_attr=(batch.edge_attr.float()
                               if batch.edge_attr is not None else None),
                    additional_text_context=batch.desc,
                )
                loss.backward()
                optimizer.step()
                total_loss += float(loss) * batch.num_graphs
                n_seen     += batch.num_graphs
        print(f"  Epoch {epoch}/3 | Train loss: {total_loss / n_seen:.4f}")


# =============================================================================
# PART 2 – Text-only RAG baseline (BM25 on linearized subgraph)
# =============================================================================

print("\n" + "=" * 60)
print("PART 2 – Text-only RAG baseline")
print("=" * 60)

from rank_bm25 import BM25Okapi


def linearize_graph(data) -> List[str]:
    """Turn a Data object into a list of `src rel dst` triples.
    In WebQSP-tiny, edge_attr is a matrix of embeddings; we use its
    row index as a placeholder for the relation name. In a real
    deployment we would store the relation label as a string."""
    triples = []
    src, dst = data.edge_index.tolist()
    for i in range(len(src)):
        triples.append(f"node_{src[i]} rel_{i} node_{dst[i]}")
    return triples


def text_only_rag(llm, data, top_k: int = 10,
                   max_out_tokens: int = 128) -> str:
    """Retrieve top-k triples by BM25 and hand them to the LLM."""
    triples = linearize_graph(data)
    corpus_tokens = [t.split() for t in triples]
    bm25 = BM25Okapi(corpus_tokens)

    q = data.question[0]
    scores = bm25.get_scores(q.split())
    top_idx = scores.argsort()[-top_k:][::-1]
    context = "\n".join(triples[i] for i in top_idx)

    prompt = (
        f"Context (graph triples):\n{context}\n\n"
        f"Question: {q}\nAnswer:"
    )
    return llm.inference([prompt], max_out_tokens=max_out_tokens)[0]


with timed("bm25_setup"):
    print("\nText-only baseline ready.")


# =============================================================================
# PART 3 – Qualitative comparison on selected questions
# =============================================================================

print("\n" + "=" * 60)
print("PART 3 – Qualitative comparison on selected questions")
print("=" * 60)

# We pick a handful of test questions. In WebQSP-tiny the reasoning types
# are not labeled; the indices below are placeholders that the reader
# should inspect and adjust. The point is not to produce a leaderboard,
# but to see side by side how the two systems behave on the same input.
categories = {
    'single_hop':  0,
    'multi_hop':   1,
    'aggregation': 2,
}

model.eval()
for name, idx in categories.items():
    if idx >= len(test_ds):
        continue
    sample = test_ds[idx]
    q = sample.question[0]
    expected = sample.label[0]

    with timed(f"qa_{name}_gretriever"):
        gr_answer = model.inference(
            question=[q],
            x=sample.x.float(),
            edge_index=sample.edge_index,
            batch=torch.zeros(sample.x.size(0), dtype=torch.long),
            edge_attr=(sample.edge_attr.float()
                       if sample.edge_attr is not None else None),
            additional_text_context=[sample.desc[0]],
            max_out_tokens=128,
        )[0]

    with timed(f"qa_{name}_textonly"):
        txt_answer = text_only_rag(llm, sample)

    print(f"\n--- {name.upper()} ---")
    print(f"Question:    {q}")
    print(f"Expected:    {expected}")
    print(f"G-Retriever: {gr_answer}")
    print(f"Text-only:   {txt_answer}")

print_timings()
print("\nDone.")