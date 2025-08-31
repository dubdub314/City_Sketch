import argparse, json
from pathlib import Path
from .data_io import list_samples, read_shapefiles, read_relations
from .graphs import build_pyg_from_layers

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--group_dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--sectors', type=int, default=8)
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--use_relations', type=int, default=1)
    ap.add_argument('--topo_tol', type=float, default=0.5)
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    samples = list_samples(args.group_dir)
    for sdir in samples:
        layers = read_shapefiles(sdir)
        rel = read_relations(sdir) if args.use_relations==1 else None
        data, meta = build_pyg_from_layers(layers['points'], layers['lines'], layers['polygons'], relations_df=rel, sectors=args.sectors, k=args.k, use_relations=bool(args.use_relations), topo_tol_units=args.topo_tol)
        nodes = meta['nodes'][['Name','Type','x','y','dir']].to_dict(orient='list')
        rec = {'sample_dir': str(sdir), 'nodes': nodes, 'sectors': args.sectors, 'use_relations': bool(args.use_relations)}
        outp = out_dir/f"{sdir.name}.json"
        outp.write_text(json.dumps(rec, ensure_ascii=False), encoding='utf-8')
        manifest.append(str(outp))
        print("Saved", outp)
    (out_dir/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print("Done. Samples:", len(manifest))

if __name__ == '__main__':
    main()
