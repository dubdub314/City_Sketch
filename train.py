import argparse, json, os
import torch, numpy as np
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm
from .dataset import GraphPairDataset
from .model import StructureFirstGNN
from .ot_matching import fgw_similarity

def nt_xent(z1, z2, temperature=0.2):
    z1 = torch.nn.functional.normalize(z1, dim=-1)
    z2 = torch.nn.functional.normalize(z2, dim=-1)
    logits = z1 @ z2.t() / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    return loss

def dense_adj_from_edge_index(edge_index, num_nodes):
    import numpy as np
    A = np.zeros((num_nodes, num_nodes), dtype=float)
    ei = edge_index.cpu().numpy()
    for s, t in ei.T:
        A[s, t] += 1.0
    A = np.maximum(A, A.T)
    A += np.eye(num_nodes)*1e-6
    return A

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--graphs_root', required=True)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--temperature', type=float, default=0.2)
    ap.add_argument('--fgw_weight', type=float, default=0.2)
    ap.add_argument('--fgw_alpha', type=float, default=0.8)
    ap.add_argument('--fgw_reg', type=float, default=1e-3)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    manifests = []
    for g in os.listdir(args.graphs_root):
        mp = os.path.join(args.graphs_root, g, 'manifest.json')
        if os.path.exists(mp):
            with open(mp,'r',encoding='utf-8') as f:
                manifests += json.load(f)
    if not manifests:
        raise FileNotFoundError('No manifest.json under graphs_root/*')
    dataset = GraphPairDataset(manifests, use_relations=True, sectors=8)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda batch: batch)

    d1, d2 = dataset[0]
    node_in = d1.x.size(1); edge_in = d1.edge_attr.size(1)
    model = StructureFirstGNN(node_in=node_in, edge_in=edge_in, hidden=args.hidden, layers=args.layers, heads=args.heads).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs+1):
        model.train(); total = 0.0
        for pairs in tqdm(loader, desc=f'Epoch {epoch}'):
            d1_list = [p[0] for p in pairs]; d2_list = [p[1] for p in pairs]
            b1 = Batch.from_data_list(d1_list).to(args.device)
            b2 = Batch.from_data_list(d2_list).to(args.device)
            h1, g1 = model(b1); h2, g2 = model(b2)
            loss = nt_xent(g1, g2, temperature=args.temperature)
            if args.fgw_weight > 0:
                fgw_losses = []
                for i in range(len(d1_list)):
                    idx1 = (b1.batch == i).nonzero(as_tuple=True)[0]
                    idx2 = (b2.batch == i).nonzero(as_tuple=True)[0]
                    H1 = h1[idx1].detach().cpu().numpy(); H2 = h2[idx2].detach().cpu().numpy()
                    A1 = dense_adj_from_edge_index(b1.edge_index[:, (b1.batch[b1.edge_index[0]]==i) & (b1.batch[b1.edge_index[1]]==i)], len(idx1))
                    A2 = dense_adj_from_edge_index(b2.edge_index[:, (b2.batch[b2.edge_index[0]]==i) & (b2.batch[b2.edge_index[1]]==i)], len(idx2))
                    sim, _ = fgw_similarity(H1, H2, A1, A2, alpha=args.fgw_alpha, reg=args.fgw_reg)
                    fgw_losses.append(1.0 - sim)
                if fgw_losses:
                    loss = loss + args.fgw_weight * (sum(fgw_losses)/len(fgw_losses))
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        print(f"Epoch {epoch}: loss={total/len(loader):.4f}")

    out = os.path.join(args.graphs_root, 'ckpt_structure_first.pt')
    torch.save(model.state_dict(), out)
    print('Saved:', out)

if __name__ == '__main__':
    main()
