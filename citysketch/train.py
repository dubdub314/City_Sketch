import argparse
from torch.utils.tensorboard import SummaryWriter, json, random
from pathlib import Path
import torch
from .utils import set_seed, CSVLogger
from torch.utils.data import DataLoader
from tqdm import tqdm
from .data_io import load_manifest
from .dataset import ContrastiveDataset, collate_contrastive
from .model import StructureFirstGNN
from .losses import nt_xent
def collect_recs(graphs_root: Path):
    recs = []
    for man in graphs_root.rglob('manifest.json'):
        recs += load_manifest(man)
    return recs

def main():
    ap = argparse.ArgumentParser()
ap.add_argument("--grad_clip", type=float, default=1.0)
ap.add_argument("--sched", choices=["none","cosine","plateau"], default="cosine")
ap.add_argument("--amp", action="store_true")

ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--log_dir", type=str, default="runs/pretrain")
ap.add_argument("--patience", type=int, default=10)

    ap.add_argument('--graphs_root', required=True)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--temperature', type=float, default=0.2)
    ap.add_argument('--fgw_weight', type=float, default=0.0)  # optional, ignored if POT not installed
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--save', default='ckpt_structure_first.pt')
    args = ap.parse_args()
    set_seed(args.seed)

    recs = collect_recs(Path(args.graphs_root))
    ds = ContrastiveDataset(recs)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_contrastive)

    # probe feature dims
    d1, d2 = ds[0]
    node_in = d1.x.size(1); edge_in = d1.edge_attr.size(1)
    model = StructureFirstGNN(node_in=node_in, edge_in=edge_in, hidden=args.hidden, layers=args.layers, heads=args.heads).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs+1):
        model.train(); total=0.0; n=0
        for B1, B2 in tqdm(loader, desc=f'Epoch {epoch}'):
            B1=B1.to(args.device); B2=B2.to(args.device)
            _, g1 = model(B1); _, g2 = model(B2)
            L = nt_xent(g1, g2, temperature=args.temperature)
            opt.zero_grad(); L.backward(); opt.step()
            total += L.item(); n+=1
        print(f'Epoch {epoch}: contrastive loss={total/max(1,n):.4f}')
    out = Path(args.graphs_root)/args.save
    torch.save(model.state_dict(), out)
    print('Saved:', out)

if __name__=='__main__':
    main()
