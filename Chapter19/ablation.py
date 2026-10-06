"""
Chapter 19 - Large Language Models Meet Graph Neural Networks
Hands-On Graph Neural Networks Using Python, Second Edition

Ablation: two models trained without the graph as text in the prompt,
evaluated on the ExplaGraphs test set, on CPU or GPU. The trained weights are
read from checkpoints/ and downloaded from Hugging Face
(giuseppefutia/hands-on-gnn-ch19-gretriever) if missing; ablation_colab.ipynb
reproduces them on a GPU. Run run.py first: Figure 19.4 also uses its results
(checkpoints/comparison.json).

PART 1  Load the ExplaGraphs test set
PART 2  Assemble the two models and load their weights
PART 3  Accuracy on the 398 test samples:
          G-Retriever trained with the GNN soft prompt only,
          LoRA trained without any graph.
        Results saved to checkpoints/ablation.json
PART 4  Figure 19.4 with the five systems, saved to figures/
"""

import csv
import json
import os
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
CHECKPOINTS = {name: f'./checkpoints/{name}_explagraphs.pt'
               for name in ['gretriever_nodesc', 'lora_nograph']}

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
# PART 2 - Assemble the two models and load their weights
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
models = {
    'gretriever_nodesc': GRetriever(llm=make_llm(), gnn=gnn, use_lora=True,
                                    mlp_out_tokens=1).to(device),
    'lora_nograph': GRetriever(llm=make_llm(), gnn=None,
                               use_lora=True).to(device),
}
for name, net in models.items():
    load_weights(net, name)


def first_word(pred):
    answer = re.sub(r'<think>.*?</think>', '', pred, flags=re.DOTALL)
    words = answer.strip().lower().split()
    return re.sub(r'[^a-z]', '', words[0]) if words else ''


def is_correct(pred, label):
    return first_word(pred).startswith(label[:4].lower())


@torch.no_grad()
def predict(system, sample):
    return models[system].inference(
        question=[sample.question],
        x=sample.x.float().to(device),
        edge_index=sample.edge_index.to(device),
        batch=torch.zeros(sample.x.size(0), dtype=torch.long, device=device),
        edge_attr=sample.edge_attr.float().to(device),
        additional_text_context=None,
        max_out_tokens=24,
    )[0]


# =============================================================================
# PART 3 - Accuracy on the test set
# =============================================================================

print('\n' + '=' * 70)
print('PART 3 - Accuracy without the graph as text')
print('=' * 70)

SYSTEMS = ['gretriever_nodesc', 'lora_nograph']
results = {s: [] for s in SYSTEMS}
records = []

for i, sample in enumerate(test_ds):
    record = {'index': i, 'label': sample.label}
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
    print(f'{s:<18}: {correct}/{n} ({100 * correct / n:.1f}%)')

print('\nPredicted labels:')
for s in SYSTEMS:
    n_support = sum(1 for r in records if r[s].startswith('supp'))
    n_counter = sum(1 for r in records if r[s].startswith('coun'))
    print(f'  {s:<18}: support {n_support}, counter {n_counter}, '
          f'other {n - n_support - n_counter}')

with open('./checkpoints/ablation.json', 'w') as f:
    json.dump({'results': results, 'records': records}, f, indent=2)


# =============================================================================
# PART 4 - Figure 19.4
# =============================================================================

print('\n' + '=' * 70)
print('PART 4 - Figure 19.4')
print('=' * 70)

with open('./checkpoints/comparison.json') as f:
    main = json.load(f)['results']
all_results = {**main, **results}
order = ['gretriever', 'no_soft_prompt', 'lora', 'gretriever_nodesc',
         'lora_nograph']
for s in order:
    correct = sum(all_results[s])
    print(f'{s:<18}: {correct}/{n} ({100 * correct / n:.1f}%)')

FONT = 'DejaVu Sans'
G0 = '#111111'; G1 = '#333333'; G2 = '#555555'
G3 = '#777777'; G4 = '#999999'; G5 = '#BBBBBB'; G6 = '#DDDDDD'
plt.rcParams['font.family'] = FONT

acc = [100 * sum(all_results[s]) / n for s in order]
names = ['G-Retriever', 'G-Retriever\nwithout\nsoft prompt', 'LoRA\nbaseline',
         'G-Retriever\nGNN only', 'LoRA\nno graph']
colors = [G1, G4, G3, G2, G5]

fig, ax = plt.subplots(figsize=(10, 4.8))
bars = ax.bar(names, acc, color=colors, edgecolor=G0, width=0.6)
for bar, value in zip(bars, acc):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 1.5, f'{value:.1f}%',
            ha='center', fontsize=11, fontweight='bold', color=G0)
chance = ax.axhline(50, color=G3, linestyle='--', linewidth=1)
ax.legend([chance], ['Chance level (balanced test set)'], loc='upper center',
          bbox_to_anchor=(0.5, -0.2), fontsize=9, frameon=False)
ax.axvline(2.5, color=G4, linewidth=0.8)
ax.text(1.0, 104, 'Graph as text in the prompt', ha='center', fontsize=10,
        color=G0)
ax.text(3.5, 104, 'No graph as text', ha='center', fontsize=10, color=G0)
ax.set_ylim(0, 110)
ax.set_yticks(range(0, 101, 20))
ax.set_ylabel('Test accuracy (%)', fontsize=10, color=G0)
ax.spines[['top', 'right']].set_visible(False)
ax.tick_params(colors=G0, labelsize=10)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig19_4_ablation.png'),
            dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('  saved figures/fig19_4_ablation.png')

print('\nDone.')