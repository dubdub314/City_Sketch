import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch_geometric.data import Batch
from .model import StructureFirstGNN
from .graphs import build_pyg_from_layers
from .accuracy_utils import match_by_name, type_consistency, per_class_scores
from .api_clients import GoogleMapsStub

def load_graphs_from_manifest(manifest_path):
    manifests = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    recs = [json.loads(Path(p).read_text(encoding='utf-8')) for p in manifests]
    datas = []
    for rec in recs:
        import geopandas as gpd, pandas as pd
        nodes = pd.DataFrame(rec['nodes'])
        pts = nodes[nodes['Type'].isin(['Landmark','Node'])].copy(); pts['geometry']=None
        ln  = nodes[nodes['Type'].isin(['Path','Edge'])].copy(); ln['geometry']=None
        pg  = nodes[nodes['Type'].isin(['District'])].copy(); pg['geometry']=None
        gdf_pts = gpd.GeoDataFrame(pts, geometry='geometry'); gdf_ln = gpd.GeoDataFrame(ln, geometry='geometry'); gdf_pg = gpd.GeoDataFrame(pg, geometry='geometry')
        d, meta = build_pyg_from_layers(gdf_pts, gdf_ln, gdf_pg, relations_df=None, sectors=rec['sectors'], k=5, use_relations=False, topo_tol_units=0.5)
        datas.append((d, meta, nodes))
    return recs, datas

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--graphs_root', required=True)
    ap.add_argument('--group', required=True)
    ap.add_argument('--real', required=False)
    ap.add_argument('--use_api', type=int, default=0)
    ap.add_argument('--api_key', type=str, default='FAKE_API_KEY')
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--weights', type=str, default='{"w_struct":0.6, "w_type":0.1, "w_topo":0.2, "w_shape":0.1}')
    args = ap.parse_args()

    weights = json.loads(args.weights)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    recG, datG = load_graphs_from_manifest(Path(args.graphs_root, args.group, 'manifest.json'))

    if args.real:
        recR, datR = load_graphs_from_manifest(Path(args.graphs_root, args.real, 'manifest.json'))
        if len(datR) != 1:
            raise RuntimeError('RealMap should have exactly one preprocessed record.')
        real_nodes = datR[0][2]
    else:
        real_nodes = None
        api = GoogleMapsStub(args.api_key)

    node_in = datG[0][0].x.size(1); edge_in = datG[0][0].edge_attr.size(1)
    model = StructureFirstGNN(node_in=node_in, edge_in=edge_in).to(args.device)
    ckpt = Path(args.graphs_root, 'ckpt_structure_first.pt')
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=args.device))
        print('Loaded checkpoint:', ckpt)
    model.eval()

    reports = []
    for rec, (data, meta, nodesG) in zip(recG, datG):
        if real_nodes is not None:
            mapping = match_by_name(nodesG, real_nodes)
            type_fid = type_consistency(nodesG, real_nodes, mapping) if mapping else 0.0
        else:
            mapping = {}
            type_fid = 0.0

        with torch.no_grad():
            bG = Batch.from_data_list([data.to(args.device)]); hG, gG = model(bG); H_G = hG.cpu().numpy()
            if real_nodes is not None:
                import geopandas as gpd
                pts = real_nodes[real_nodes['Type'].isin(['Landmark','Node'])].copy(); pts['geometry']=None
                ln  = real_nodes[real_nodes['Type'].isin(['Path','Edge'])].copy(); ln['geometry']=None
                pg  = real_nodes[real_nodes['Type'].isin(['District'])].copy(); pg['geometry']=None
                gdf_pts = gpd.GeoDataFrame(pts, geometry='geometry'); gdf_ln = gpd.GeoDataFrame(ln, geometry='geometry'); gdf_pg = gpd.GeoDataFrame(pg, geometry='geometry')
                dR, metaR = build_pyg_from_layers(gdf_pts, gdf_ln, gdf_pg, relations_df=None, sectors=rec['sectors'], k=5, use_relations=False, topo_tol_units=0.5)
                bR = Batch.from_data_list([dR.to(args.device)]); hR, gR = model(bR); H_R = hR.cpu().numpy()
                from .ot_matching import hungarian_similarity
                score, coverage, pairs, sims = hungarian_similarity(H_G, H_R)
                struct_fid = float(score)
                # per-geometry optional matrix using sims
                per_geom = None
            else:
                struct_fid = float(0.0); per_geom=None

        topo_fid = 0.0; shape_fid = 0.0
        # NOTE: Full topology/shape accuracy requires RealMap geometries; here we placeholder as type-based proxy if mapping exists
        if real_nodes is not None and mapping:
            broadA = nodesG['Type'].map(lambda t: 'pt' if t in ['Landmark','Node'] else ('ln' if t in ['Path','Edge'] else 'pg'))
            broadB = real_nodes['Type'].map(lambda t: 'pt' if t in ['Landmark','Node'] else ('ln' if t in ['Path','Edge'] else 'pg'))
            same_broad = [1 for i,j in mapping.items() if broadA.iloc[i]==broadB.iloc[j]]
            topo_fid = len(same_broad)/max(1,len(mapping))
            shape_fid = topo_fid

        w = weights
        fidelity = w['w_struct']*struct_fid + w['w_type']*type_fid + w['w_topo']*topo_fid + w['w_shape']*shape_fid
        reports.append({'sample': Path(rec['sample_dir']).name, 'fidelity': float(fidelity),
                        'structure': float(struct_fid), 'type': float(type_fid), 'topology': float(topo_fid), 'shape': float(shape_fid)})

    outp = Path(args.out)/'accuracy_report.json'
    outp.write_text(json.dumps({'reports': reports, 'weights': weights}, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Saved:', outp)

if __name__ == '__main__':
    main()
