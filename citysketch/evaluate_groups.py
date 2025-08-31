import argparse
import json, numpy as np, torch
from .ot_matching import fgw_distance, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch_geometric.data import Batch
from .model import StructureFirstGNN, PairwiseHead
from .graphs import build_pyg_from_layers

def load_group_graphs(graphs_root_group_path):
    manp = Path(graphs_root_group_path)/'manifest.json'
    manifests = json.loads(manp.read_text(encoding='utf-8'))
    recs = [json.loads(Path(p).read_text(encoding='utf-8')) for p in manifests]
    datas = []
    for rec in recs:
        import geopandas as gpd, pandas as pd
        nodes = pd.DataFrame(rec['nodes'])
        pts = nodes[nodes['Type'].isin(['Landmark','Node'])].copy(); pts['geometry']=None
        ln  = nodes[nodes['Type'].isin(['Path','Edge'])].copy(); ln['geometry']=None
        pg  = nodes[nodes['Type'].isin(['District'])].copy(); pg['geometry']=None
        gdf_pts = gpd.GeoDataFrame(pts, geometry='geometry'); gdf_ln = gpd.GeoDataFrame(ln, geometry='geometry'); gdf_pg = gpd.GeoDataFrame(pg, geometry='geometry')
        d, _ = build_pyg_from_layers(gdf_pts, gdf_ln, gdf_pg, relations_df=None, sectors=rec['sectors'], k=5, use_relations=True, topo_tol_units=0.5)
        datas.append(d)
    return recs, datas

def load_best_model(graphs_root: Path, node_in, edge_in, device):
    sup = graphs_root/'ckpt_sup.pt'
    unsup = graphs_root/'ckpt_structure_first.pt'
    model = StructureFirstGNN(node_in=node_in, edge_in=edge_in).to(device)
    head = None
    if sup.exists():
        ckpt = torch.load(sup, map_location=device)
        model.load_state_dict(ckpt['backbone'])
        head = PairwiseHead(dim=model.readout[-1].out_features if hasattr(model.readout[-1],'out_features') else 128).to(device)
        head.load_state_dict(ckpt['head'])
        print('Loaded supervised checkpoint:', sup)
    elif unsup.exists():
        model.load_state_dict(torch.load(unsup, map_location=device))
        print('Loaded unsupervised checkpoint:', unsup)
    else:
        print('No checkpoint found; using randomly initialized model (not recommended).')
    return model, head

def main():
    ap = argparse.ArgumentParser()
ap.add_argument("--use_fgw", action="store_true")
ap.add_argument("--topk", type=int, default=20)

    ap.add_argument('--graphs_root', required=True)
    ap.add_argument('--group_a', required=True)
    ap.add_argument('--group_b', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    root = Path(args.graphs_root)
    recA, dataA = load_group_graphs(root/args.group_a)
    recB, dataB = load_group_graphs(root/args.group_b)

    node_in = dataA[0].x.size(1); edge_in = dataA[0].edge_attr.size(1)
    model, head = load_best_model(root, node_in, edge_in, args.device)

    model.eval()
    with torch.no_grad():
        GA, GB = [], []
        for d in dataA:
            b = Batch.from_data_list([d.to(args.device)]); _, g = model(b); GA.append(g.squeeze(0).cpu().numpy())
        for d in dataB:
            b = Batch.from_data_list([d.to(args.device)]); _, g = model(b); GB.append(g.squeeze(0).cpu().numpy())
        GA = np.stack(GA); GB = np.stack(GB)

    def cos(u,v):
        return (u@v)/(np.linalg.norm(u)*np.linalg.norm(v)+1e-9)
    M = np.zeros((len(GA), len(GB)))
    for i in range(len(GA)):
        for j in range(len(GB)):
            M[i,j] = (cos(GA[i], GB[j]) + 1)/2

    from scipy.optimize import linear_sum_assignment
    C = 1 - M
    ri, cj = linear_sum_assignment(C)
    group_fidelity = float(1 - C[ri, cj].mean())
    pairing = [(Path(recA[i]['sample_dir']).name, Path(recB[j]['sample_dir']).name, float(M[i,j])) for i,j in zip(ri,cj)]

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(M, index=[Path(r['sample_dir']).name for r in recA], columns=[Path(r['sample_dir']).name for r in recB]).to_csv(out_dir/'similarity_matrix.csv', encoding='utf-8-sig')
    (out_dir/'report.json').write_text(json.dumps({'group_fidelity': group_fidelity, 'pairing': pairing}, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Saved:', out_dir)

if __name__ == '__main__':
    main()

def _write_topk_mismatches(args, T, A, B, na, nb, topk=20):
    """Use FGW coupling to extract weakly matched nodes as error candidates."""
    import numpy as np, csv, os, torch
    # Build node name lists
    names_a=[]; for a in A: names_a += getattr(torch.load(a["path"], map_location="cpu"), "names", [f"a_{i}" for i in range(len(T))])
    names_b=[]; for b in B: names_b += getattr(torch.load(b["path"], map_location="cpu"), "names", [f"b_{j}" for j in range(len(T[0]))])
    # Low coupling rows/cols
    row_best = T.max(axis=1); col_best = T.max(axis=0)
    i_sorted = np.argsort(row_best)[:topk]
    j_sorted = np.argsort(col_best)[:topk]
    with open(os.path.join(args.out, "fgw_topk.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["side","index","name","best_coupling"])
        for i in i_sorted: w.writerow(["A", int(i), names_a[i] if i<len(names_a) else f"A_{i}", float(row_best[i])])
        for j in j_sorted: w.writerow(["B", int(j), names_b[j] if j<len(names_b) else f"B_{j}", float(col_best[j])])
