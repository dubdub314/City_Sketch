import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import softmax as pyg_softmax
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
        # attention logits per head
        logits = (Q_i * K_j).sum(dim=-1) + (Q_i * E).sum(dim=-1)  # [E, H]
        # per-head incoming-edge softmax
        alphas = []
        for h in range(logits.size(1)):
            alphas.append(pyg_softmax(F.leaky_relu(logits[:,h], negative_slope=0.2), index))
        alpha = torch.stack(alphas, dim=1)  # [E, H]
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
        # Assume edge_attr = [d,cos,sin] + sectorOH(S) + etypeOH(T)
        S = 8  # default sectors
        K = edge_attr.size(1) - (3 + S)
        if K <= 0:
            S = 16; K = edge_attr.size(1) - (3 + S)
            if K <= 0: K = 8  # fallback
        etoh = edge_attr[:, -K:]
        # Build masks by name if we know category size; else heuristics by position
        # Try to map names to indices via EDGE_CAT
        def has_tag(tag):
            try:
                idx = EDGE_CAT.index(tag)
                return etoh[:, idx] > 0.5
            except Exception:
                return None
        m_hyp_dp = has_tag("HYP_DISTRICT_PATH")
        m_hyp_pn = has_tag("HYP_PATH_NODE")
        # topo: any tag starting with TOPO_
        m_topo = None
        try:
            topo_idx = [i for i,t in enumerate(EDGE_CAT) if t.startswith("TOPO_")]
            if topo_idx:
                m_topo = etoh[:, topo_idx].sum(dim=1) > 0
        except Exception:
            m_topo = None
        # ssg/rel: others
        used = torch.zeros_like(etoh[:,0], dtype=torch.bool)
        for m in [m_hyp_dp, m_hyp_pn, m_topo]:
            if m is not None: used = used | m
        m_other = ~used

        for conv, norm in zip(self.convs, self.norms):
            # Stage 0: hyper District↔Path
            if m_hyp_dp is not None and m_hyp_dp.any():
                h = norm(h + conv(h, edge_index[:, m_hyp_dp], edge_attr[m_hyp_dp]))
                h = F.dropout(h, p=self.dropout, training=self.training)
            # Stage 1: hyper Path↔Node
            if m_hyp_pn is not None and m_hyp_pn.any():
                h = norm(h + conv(h, edge_index[:, m_hyp_pn], edge_attr[m_hyp_pn]))
                h = F.dropout(h, p=self.dropout, training=self.training)
            # Stage 2: topology edges
            if m_topo is not None and m_topo.any():
                h = norm(h + conv(h, edge_index[:, m_topo], edge_attr[m_topo]))
                h = F.dropout(h, p=self.dropout, training=self.training)
            # Stage 3: remaining (SSG/REL)
            if m_other.any():
                h = norm(h + conv(h, edge_index[:, m_other], edge_attr[m_other]))
                h = F.dropout(h, p=self.dropout, training=self.training)
        g = self.readout(global_mean_pool(h, batch))
        return h, g



class PairwiseHead(nn.Module):
    def __init__(self, dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim*4, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden//2), nn.ReLU(),
            nn.Linear(hidden//2, 1)
        )
    def forward(self, ga, gb):
        z = torch.cat([ga, gb, torch.abs(ga-gb), ga*gb], dim=-1)
        logit = self.net(z).squeeze(-1)
        return logit