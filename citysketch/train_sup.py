import argparse, json
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
from torch_geometric.data import Batch
from .model import StructureFirstGNN, PairwiseHead
from .losses import nt_xent, supcon_loss, pair_loss
from .data_io import load_manifest

def load_manifests_dict(graphs_root: Path):
    d = {}
    for man in graphs_root.rglob('manifest.json'):
        recs = load_manifest(man)
        for r in recs:
            d[Path(r['sample_dir']).name] = r
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--graphs_root', required=True)
    ap.add_argument('--pairs_csv', required=True)
    ap.add_argument('--pair_loss', default='bce', choices=['bce','mse','margin'])
    ap.add_argument('--losses', default='{"w_contrast":0.5, "w_pair":1.0, "w_supcon":0.0}')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--temperature', type=float, default=0.2)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--save', default='ckpt_sup.pt')
    args = ap.parse_args()

    weights = json.loads(args.losses)
    graphs_root = Path(args.graphs_root)
    manifest_dict = load_manifests_dict(graphs_root)

    pairs_df = pd.read_csv(args.pairs_csv)
    pairs_df.columns = [c.strip().lower() for c in pairs_df.columns]
    if 'weight' not in pairs_df.columns:
        pairs_df['weight'] = 1.0
    for col in ['sample_a','sample_b','label']:
        if col not in pairs_df.columns:
            raise ValueError(f'Missing column: {col}')

    from .dataset_sup import PairwiseDataset, collate_pairs
    ds = PairwiseDataset(manifest_dict, pairs_df)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_pairs)

    d1, d2, _, _ = ds[0]
    node_in = d1.x.size(1); edge_in = d1.edge_attr.size(1)
    model = StructureFirstGNN(node_in=node_in, edge_in=edge_in, hidden=args.hidden, layers=args.layers, heads=args.heads).to(args.device)
    head = PairwiseHead(dim=args.hidden).to(args.device)
    opt = torch.optim.Adam(list(model.parameters())+list(head.parameters()), lr=args.lr)

    for epoch in range(1, args.epochs+1):
        model.train(); head.train(); total_pair=0.0; n=0
        for Ba, Bb, y, w in tqdm(loader, desc=f'Epoch {epoch}'):
            Ba=Ba.to(args.device); Bb=Bb.to(args.device); y=y.to(args.device); w=w.to(args.device)
            ha, ga = model(Ba); hb, gb = model(Bb)
            # pair loss
            logit = head(ga, gb)
            L_pair = pair_loss(logit, y, kind=args.pair_loss)
            # optional contrast
            L_contrast = nt_xent(ga, gb, temperature=args.temperature) if weights.get('w_contrast',0)>0 else 0.0
            # optional supervised contrast (weak, by thresholding label in batch)
            if weights.get('w_supcon',0)>0:
                labels = (y>0.5).long()
                feats = torch.cat([ga, gb], dim=0)
                labs = torch.cat([labels, labels], dim=0)
                L_supcon = supcon_loss(feats, labs, temperature=args.temperature)
            else:
                L_supcon = 0.0

            L = weights.get('w_pair',1.0)*L_pair + weights.get('w_contrast',0.0)*L_contrast + weights.get('w_supcon',0.0)*L_supcon
            opt.zero_grad(); 
            if isinstance(L, float):
                torch.tensor(L, dtype=torch.float32, device=args.device).backward()
            else:
                L.backward()
            opt.step()
            total_pair += (L_pair if not isinstance(L_pair,float) else torch.tensor(L_pair)).item(); n+=1
        print(f'Epoch {epoch}: pair_loss={total_pair/max(1,n):.4f}')
    out = graphs_root/args.save
    torch.save({'backbone': model.state_dict(), 'head': head.state_dict()}, out)
    print('Saved:', out)

if __name__ == '__main__':
    main()
