"""
One-off utility (not part of the repository).
Reads the full checkpoint saved by train_colab.ipynb and keeps only the
trainable weights of G-Retriever (LoRA adapters, GAT encoder, projector).
"""

import os

import torch
from torch_geometric.llm.models import LLM, GRetriever
from torch_geometric.nn import GAT

FULL = './checkpoints/epoch_10.pt'
SLIM = './checkpoints/gretriever_explagraphs.pt'

llm = LLM(model_name='Qwen/Qwen3-0.6B', num_params=0.6, dtype=torch.bfloat16)
gnn = GAT(in_channels=1024, hidden_channels=1024, num_layers=4,
          out_channels=1024, heads=4)
model = GRetriever(llm=llm, gnn=gnn, use_lora=True, mlp_out_tokens=1)

trainable = {name for name, p in model.named_parameters() if p.requires_grad}

ckpt = torch.load(FULL, map_location='cpu')
state = {k: v for k, v in ckpt['state_dict'].items() if k in trainable}
missing = trainable - set(state)

print(f'Full checkpoint: {len(ckpt["state_dict"])} tensors')
print(f'Trainable in model: {len(trainable)} tensors, '
      f'{sum(model.get_parameter(n).numel() for n in trainable)/1e6:.1f}M parameters')
print(f'Kept: {len(state)} tensors, missing: {len(missing)}')

torch.save({'epoch': ckpt['epoch'], 'val_acc': ckpt['val_acc'],
            'state_dict': state}, SLIM)
print(f'Saved {SLIM} ({os.path.getsize(SLIM)/1e6:.1f} MB)')