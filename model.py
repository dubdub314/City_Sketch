import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.nn.inits import glorot, zeros

class EdgeAwareGAT(MessagePassing):
    def __init__(self, in_dim, out_dim, edge_dim, heads=4, dropout=0.0):
        super().__init__(aggr='add', node_dim=0)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.dropout = dropout
        self.lin_q = nn.Linear(in_dim, heads*out_dim, bias=False)
        self.lin_k = nn.Linear(in_dim, heads*out_dim, bias=False)
        self.lin_v = nn.Linear(in_dim, heads*out_dim, bias=False)
        self.lin_e = nn.Linear(edge_dim, heads*out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(heads*out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_q.weight); glorot(self.lin_k.weight); glorot(self.lin_v.weight); glorot(self.lin_e.weight)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        H, D = self.heads, self.out_dim
        Q = self.lin_q(x).view(-1, H, D)
        K = self.lin_k(x).view(-1, H, D)
        V = self.lin_v(x).view(-1, H, D)
        E = self.lin_e(edge_attr).view(-1, H, D)
        out = self.propagate(edge_index, Q=Q, K=K, V=V, E=E, size=None)
        out = out + self.bias
        return out.reshape(-1, H*D)

    def message(self, Q_i, K_j, V_j, E, index, ptr, size_i):
        logits = (Q_i * K_j).sum(dim=-1) + (Q_i * E).sum(dim=-1)
        alpha = F.leaky_relu(logits, negative_slope=0.2)
        alpha = torch.softmax(alpha, dim=0)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return V_j * alpha.unsqueeze(-1)

class StructureFirstGNN(nn.Module):
    def __init__(self, node_in, edge_in, hidden=128, layers=3, heads=4, dropout=0.1):
        super().__init__()
        self.enc = nn.Linear(node_in, hidden)
        self.convs = nn.ModuleList([EdgeAwareGAT(hidden, hidden//heads, edge_in, heads=heads, dropout=dropout) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.dropout = dropout
        self.readout = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        h = torch.relu(self.enc(x))
        for conv, norm in zip(self.convs, self.norms):
            h = norm(h + conv(h, edge_index, edge_attr))
            h = F.dropout(h, p=self.dropout, training=self.training)
        g = self.readout(global_mean_pool(h, batch))
        return h, g
