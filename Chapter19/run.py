"""
Chapter 19 – Large Language Models Meet Graph Neural Networks
Hands-On Graph Neural Networks Using Python, Second Edition

Evaluation of G-Retriever on the ExplaGraphs test set against two baselines,
on CPU or GPU. The trained weights are read from checkpoints/ and downloaded
from Hugging Face (giuseppefutia/hands-on-gnn-ch19-gretriever) if missing;
train_colab.ipynb reproduces them on a GPU.

PART 1  Load the ExplaGraphs test set
PART 2  Assemble G-Retriever and the LoRA baseline, load their weights
PART 3  Predict one test sample with G-Retriever
PART 4  Accuracy of three systems on the 398 test samples:
          G-Retriever (soft prompt + linearized graph),
          G-Retriever without the soft prompt (same prompt, inference only),
          LoRA baseline trained without GNN (same prompt).
        Results saved to checkpoints/comparison.json
PART 5  Figure 19.3 from the comparison, saved to figures/
"""

import csv
import json
import os
import sys
import random
import re
import urllib.request

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from huggingface_hub import hf_hub_download
from torch_geometric.data import Data
from torch_geometric.llm.models import LLM, GRetriever, SentenceTransformer
from torch_geometric.nn import GAT

HF_REPO = 'giuseppefutia/hands-on-gnn-ch19-gretriever'
CHECKPOINTS = {'gretriever': './checkpoints/gretriever_explagraphs.pt',
               'lora': './checkpoints/lora_explagraphs.pt'}

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"}')

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)


# =============================================================================
# PART 1 - Load the ExplaGraphs test set
# =============================================================================

print('\n' + '=' * 70)
print('PART 1 - Load the ExplaGraphs test set')
print('=' * 70)

DATA_DIR = './data/ExplaGraphs'
BASE = 'https://raw.githubusercontent.com/swarnaHub/ExplaGraphs/master/data'
os.makedirs(DATA_DIR, exist_ok=True)


def download_files():
    for name in ['train.tsv', 'dev.tsv']:
        local = os.path.join(DATA_DIR, name)
        if not os.path.exists(local):
            urllib.request.urlretrieve(f'{BASE}/{name}', local)


def parse_graph(graph_str):
    triples = []
    for match in re.findall(r'\(([^)]+)\)', graph_str):
        parts = [p.strip() for p in match.split(';')]
        if len(parts) == 3:
            triples.append(tuple(parts))
    return triples


def read_tsv(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for r in reader:
            if len(r) < 4:
                continue
            belief, argument, stance, graph_str = r[:4]
            if stance not in ('support', 'counter'):
                continue
            triples = parse_graph(graph_str)
            if not triples:
                continue
            rows.append({'belief': belief.strip(),
                         'argument': argument.strip(),
                         'stance': stance.strip(),
                         'triples': triples})
    return rows


def build_dataset(rows, sbert):
    dataset = []
    for row in rows:
        nodes, node_ids, es, ed, er = [], {}, [], [], []
        for src, rel, dst in row['triples']:
            for txt in (src, dst):
                if txt not in node_ids:
                    node_ids[txt] = len(nodes)
                    nodes.append(txt)
            es.append(node_ids[src])
            ed.append(node_ids[dst])
            er.append(rel)

        x = sbert.encode(nodes, batch_size=64, output_device='cpu')
        edge_attr = sbert.encode(er, batch_size=64, output_device='cpu')

        s = torch.tensor(es, dtype=torch.long)
        d = torch.tensor(ed, dtype=torch.long)
        edge_index = torch.stack([torch.cat([s, d]), torch.cat([d, s])], dim=0)
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)

        desc = ' '.join(f'({s}; {r}; {d})' for s, r, d in row['triples'])
        question = (f"/no_think Belief: {row['belief']}\n"
                    f"Argument: {row['argument']}\n"
                    f"Does the argument support or counter the belief? "
                    f"Answer with a single word: support or counter.")

        dataset.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                            question=question, label=row['stance'], desc=desc))
    return dataset


download_files()
sbert = SentenceTransformer(
    'sentence-transformers/all-roberta-large-v1'
).to(device)
sbert.eval()

test_rows = read_tsv(os.path.join(DATA_DIR, 'dev.tsv'))
test_ds = build_dataset(test_rows, sbert)
support = sum(d.label == 'support' for d in test_ds)
print(f'test={len(test_ds)} (support: {support}, counter: {len(test_ds) - support})')


# =============================================================================
# PART 2 - Assemble G-Retriever and the LoRA baseline, load their weights
# =============================================================================

print('\n' + '=' * 70)
print('PART 2 - Assemble the models and load their weights')
print('=' * 70)


def make_llm():
    llm = LLM(
        model_name='Qwen/Qwen3-0.6B',
        num_params=0.6,
        dtype=torch.bfloat16,
        sys_prompt=(
            '/no_think You are a helpful assistant. Given a belief, an '
            'argument, and a commonsense graph, decide whether the argument '
            'supports or counters the belief. Answer with a single word: '
            'support or counter.'
        ),
    )
    llm.llm.generation_config.do_sample = False
    llm.llm.generation_config.temperature = None
    llm.llm.generation_config.top_p = None
    llm.llm.generation_config.top_k = None
    return llm


def load_weights(model, name):
    path = CHECKPOINTS[name]
    if not os.path.exists(path):
        hf_hub_download(repo_id=HF_REPO, filename=os.path.basename(path),
                        local_dir=os.path.dirname(path))
    ckpt = torch.load(path, map_location='cpu')
    result = model.load_state_dict(ckpt['state_dict'], strict=False)
    print(f'{name}: loaded {len(ckpt["state_dict"])} tensors from {path} '
          f'(val_acc={ckpt["val_acc"]:.3f}), '
          f'unexpected keys: {len(result.unexpected_keys)}')
    model.eval()


gnn = GAT(
    in_channels=1024,
    hidden_channels=1024,
    num_layers=4,
    out_channels=1024,
    heads=4,
    edge_dim=1024,
)
model = GRetriever(llm=make_llm(), gnn=gnn, use_lora=True,
                   mlp_out_tokens=1).to(device)
baseline = GRetriever(llm=make_llm(), gnn=None, use_lora=True).to(device)

load_weights(model, 'gretriever')
load_weights(baseline, 'lora')


def first_word(pred):
    answer = re.sub(r'<think>.*?</think>', '', pred, flags=re.DOTALL)
    words = answer.strip().lower().split()
    return re.sub(r'[^a-z]', '', words[0]) if words else ''


def is_correct(pred, label):
    return first_word(pred).startswith(label[:4].lower())


@torch.no_grad()
def predict(system, sample):
    if system == 'no_soft_prompt':
        return model.llm.inference(question=[sample.question],
                                   context=[sample.desc], max_tokens=24)[0]
    net = model if system == 'gretriever' else baseline
    return net.inference(
        question=[sample.question],
        x=sample.x.float().to(device),
        edge_index=sample.edge_index.to(device),
        batch=torch.zeros(sample.x.size(0), dtype=torch.long, device=device),
        edge_attr=sample.edge_attr.float().to(device),
        additional_text_context=[sample.desc],
        max_out_tokens=24,
    )[0]


# =============================================================================
# PART 3 - Predict one test sample with G-Retriever
# =============================================================================

print('\n' + '=' * 70)
print('PART 3 - Predict one test sample')
print('=' * 70)

sample = test_ds[0]
pred = predict('gretriever', sample)
answer = re.sub(r'<think>.*?</think>', '', pred, flags=re.DOTALL).strip()
print(f'Expected:  {sample.label}')
print(f'Predicted: {answer}')


# =============================================================================
# PART 4 - Accuracy of the three systems on the test set
# =============================================================================

print('\n' + '=' * 70)
print('PART 4 - G-Retriever vs baselines on the test set')
print('=' * 70)

SYSTEMS = ['gretriever', 'no_soft_prompt', 'lora']
results = {s: [] for s in SYSTEMS}
records = []

for i, sample in enumerate(test_ds):
    record = {'index': i,
              'belief': test_rows[i]['belief'],
              'argument': test_rows[i]['argument'],
              'triples': sample.desc,
              'label': sample.label}
    for s in SYSTEMS:
        raw = predict(s, sample)
        results[s].append(is_correct(raw, sample.label))
        record[s] = first_word(raw)
    records.append(record)
    if (i + 1) % 50 == 0:
        print(f'  {i + 1}/{len(test_ds)} samples')

n = len(test_ds)
print()
for s in SYSTEMS:
    correct = sum(results[s])
    print(f'{s:<15}: {correct}/{n} ({100 * correct / n:.1f}%)')

g, b = results['gretriever'], results['lora']
both = sum(1 for i in range(n) if g[i] and b[i])
g_only = sum(1 for i in range(n) if g[i] and not b[i])
b_only = sum(1 for i in range(n) if not g[i] and b[i])
neither = sum(1 for i in range(n) if not g[i] and not b[i])

print('\nG-Retriever vs LoRA baseline:')
print(f'Both correct:       {both}')
print(f'G-Retriever only:   {g_only}')
print(f'LoRA baseline only: {b_only}')
print(f'Both wrong:         {neither}')

with open('./checkpoints/comparison.json', 'w') as f:
    json.dump({'results': results, 'records': records}, f, indent=2)


# =============================================================================
# PART 5 - Figure 19.3
# =============================================================================

print('\n' + '=' * 70)
print('PART 5 - Figure 19.3')
print('=' * 70)

FONT = 'DejaVu Sans'
sys.path.insert(0, os.path.join(HERE, '..', 'tools'))
from bookstyle import (apply, figure, save as save_fig, INK, INK_SOFT,
                       G3 as BG3, G4 as BG4, G5 as BG5, G6 as BG6)

apply()
G0 = '#111111'

acc = [100 * sum(results[s]) / n for s in SYSTEMS]
names = ['G-Retriever', 'G-Retriever\nwithout soft prompt', 'LoRA baseline\n(no GNN)']
counts = [both, g_only, b_only, neither]

# Stacked, not side by side: at 5.5 inches the right panel's category
# labels would take most of its 2.7.
fig, axes = figure(aspect=0.95, nrows=2)

ax = axes[0]
bars = ax.bar(names, acc, color=[BG4, BG6, BG5], edgecolor=BG3, width=0.55)
for bar, value in zip(bars, acc):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 1.5, f'{value:.1f}%',
            ha='center', fontsize=8.5, fontweight='bold', color=INK)
ax.set_ylim(0, 100)
ax.set_ylabel('Test accuracy (%)', fontsize=9, color=INK)
ax.set_title(f'Accuracy on {n} test samples', fontsize=9.5, color=INK,
             fontweight='bold')

ax = axes[1]
labels = ['Both correct', 'G-Retriever only', 'LoRA baseline only',
          'Both wrong']
bars = ax.barh(labels[::-1], counts[::-1], color=[BG6, BG5, BG4, BG6][::-1],
               edgecolor=BG3, height=0.55)
for bar, value in zip(bars, counts[::-1]):
    ax.text(bar.get_width() + 3, bar.get_y() + bar.get_height() / 2,
            str(value), va='center', fontsize=8.5, fontweight='bold',
            color=INK)
ax.set_xlim(0, max(counts) * 1.15)
ax.set_xlabel('Number of test samples', fontsize=9, color=INK)
ax.set_title('G-Retriever vs LoRA baseline', fontsize=9.5, color=INK,
             fontweight='bold')

for ax in axes:
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(colors=INK, labelsize=8.5)

save_fig(fig, os.path.join(FIG_DIR, 'fig19_3_gretriever_vs_textonly.png'))
print('  saved figures/fig19_3_gretriever_vs_textonly.png')

print('\nDone.')