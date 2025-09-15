import argparse, json
from pathlib import Path
from .data_io import read_realmap
from .graphs import build_pyg_from_layers

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--real_dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--sectors', type=int, default=8)
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--topo_tol', type=float, default=0.5)
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    layers = read_realmap(args.real_dir)
    data, meta = build_pyg_from_layers(layers['points'], layers['lines'], layers['polygons'], relations_df=None, sectors=args.sectors, k=args.k, use_relations=False, topo_tol_units=args.topo_tol)
    nodes = meta['nodes'][['Name','Type','x','y','dir']].to_dict(orient='list')
    rec = {'sample_dir': str(args.real_dir), 'nodes': nodes, 'sectors': args.sectors, 'use_relations': False}
    outp = out_dir/'RealMap.json'
    outp.write_text(json.dumps(rec, ensure_ascii=False), encoding='utf-8')
    (out_dir/'manifest.json').write_text(json.dumps([str(outp)], indent=2), encoding='utf-8')
    print("Saved", outp)

if __name__ == '__main__':
    main()
