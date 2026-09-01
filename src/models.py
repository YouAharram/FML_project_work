"""Architetture a grafo: GIN, GINE e GCN con la stessa struttura di contorno. """
import torch
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch import nn
from torch_geometric.nn import (
    GCNConv,
    GINConv,
    GINEConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

POOLING = {"mean": global_mean_pool, "sum": global_add_pool, "max": global_max_pool}
CONVS = ("gin", "gine", "gcn")


def _mlp(hidden):
    return nn.Sequential(
        nn.Linear(hidden, 2 * hidden), nn.BatchNorm1d(2 * hidden), nn.ReLU(),
        nn.Linear(2 * hidden, hidden),
    )


class GNN(nn.Module):
    def __init__(self, conv="gin", num_layers=5, hidden=300, dropout=0.5,
                 pooling="mean", num_tasks=12):
        super().__init__()
        if conv not in CONVS:
            raise ValueError(f"conv sconosciuto: {conv}")
        if pooling not in POOLING:
            raise ValueError(f"pooling sconosciuto: {pooling}")
        if num_layers < 1:
            raise ValueError("serve almeno uno strato")

        self.conv_type = conv
        self.num_layers = num_layers
        self.dropout = dropout
        self.pool = POOLING[pooling]

        self.atom_encoder = AtomEncoder(hidden)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.bond_encoders = nn.ModuleList() if conv == "gine" else None

        for _ in range(num_layers):
            if conv == "gin":
                self.convs.append(GINConv(_mlp(hidden), train_eps=True))
            elif conv == "gine":
                self.convs.append(GINEConv(_mlp(hidden), train_eps=True))
                self.bond_encoders.append(BondEncoder(hidden))
            else:
                self.convs.append(GCNConv(hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))

        self.head = nn.Linear(hidden, num_tasks)

    def forward(self, data):
        h = self.atom_encoder(data.x)

        for i, conv in enumerate(self.convs):
            if self.conv_type == "gine":
                h = conv(h, data.edge_index, self.bond_encoders[i](data.edge_attr))
            else:
                h = conv(h, data.edge_index)
            h = self.bns[i](h)
            if i < self.num_layers - 1:
                h = F.relu(h)
            h = F.dropout(h, self.dropout, training=self.training)

        return self.head(self.pool(h, data.batch))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
