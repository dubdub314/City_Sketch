import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .data_io import load_manifest
from .dataset_sup import load_graph_from_rec
from .model import StructureFirstGNN, PairwiseHead

def build_embedder(ckpt_path: Path, graphs_root: Path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    any_manifest = next((p for p in graphs_root.rglob('manifest.json')), None)
    recs = load_manifest(any_manifest)
    d0, _ = load_graph_from_rec(recs[0], use_relations=False)
    node_in = d0.x.size(1); edge_in = d0.edge_attr.size(1)
    model = StructureFirstGNN(node_in=node_in, edge_in=edge_in).to(device)
    head = PairwiseHead(dim=model.readout[-1].out_features if hasattr(model.readout[-1],'out_features') else 128).to(device)
    model.load_state_dict(ckpt['backbone']); head.load_state_dict(ckpt['head'])
    model.eval(); head.eval()
    return model, head

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--graphs_root', required=True)
    ap.add_argument('--pairs_csv', required=True)   # validation pairs with labels in [0,1]
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', default='calibrator.json')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    graphs_root = Path(args.graphs_root)
    pairs = pd.read_csv(args.pairs_csv)
    pairs.columns = [c.strip().lower() for c in pairs.columns]
    model, head = build_embedder(Path(args.ckpt), graphs_root, args.device)

    # Build mapping sample_id -> rec
    rec_map = {}
    for man in graphs_root.rglob('manifest.json'):
        for rec in load_manifest(man):
            rec_map[Path(rec['sample_dir']).name] = rec

    logits = []; labels = []
    with torch.no_grad():
        for _, row in pairs.iterrows():
            ra = rec_map[row['sample_a']]; rb = rec_map[row['sample_b']]
            da, _ = load_graph_from_rec(ra, use_relations=False)
            db, _ = load_graph_from_rec(rb, use_relations=False)
            Ba = Batch.from_data_list([da.to(args.device)]); Bb=Batch.from_data_list([db.to(args.device)])
            _, ga = model(Ba); _, gb = model(Bb)
            logit = head(ga, gb).squeeze().cpu().item()
            logits.append(logit); labels.append(float(row['label']))
    logits = np.array(logits); labels = np.array(labels)

    lr = LogisticRegression(max_iter=1000).fit(logits.reshape(-1,1), labels>0.5)
    iso = IsotonicRegression(out_of_bounds='clip').fit(logits, labels)

    out = {
        'platt_coef': lr.coef_.ravel().tolist(),
        'platt_intercept': lr.intercept_.ravel().tolist(),
        'isotonic_x': iso.X_thresholds_.tolist(),
        'isotonic_y': iso.y_thresholds_.tolist()
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding='utf-8')
    print('Saved calibrator to', args.out)

if __name__ == '__main__':
    main()
