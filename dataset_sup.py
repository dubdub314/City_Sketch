import pandas as pd, geopandas as gpd
from pathlib import Path
import torch
from .graphs import build_pyg_from_layers

def load_graph_from_rec(rec, use_relations=False):
    import pandas as pd
    nodes = pd.DataFrame(rec['nodes'])
    pts = nodes[nodes['Type'].isin(['Landmark','Node'])].copy(); pts['geometry']=None
    ln  = nodes[nodes['Type'].isin(['Path','Edge'])].copy(); ln['geometry']=None
    pg  = nodes[nodes['Type'].isin(['District'])].copy(); pg['geometry']=None
    gdf_pts = gpd.GeoDataFrame(pts, geometry='geometry'); gdf_ln = gpd.GeoDataFrame(ln, geometry='geometry'); gdf_pg = gpd.GeoDataFrame(pg, geometry='geometry')
    d, _ = build_pyg_from_layers(gdf_pts, gdf_ln, gdf_pg, relations_df=None, sectors=rec['sectors'], k=5, use_relations=use_relations, topo_tol_units=0.5)
    return d, nodes

class PairwiseDataset(torch.utils.data.Dataset):
    def __init__(self, manifests_by_id: dict, pairs_df: pd.DataFrame):
        self.m = manifests_by_id
        self.df = pairs_df.reset_index(drop=True)
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ra = self.m[row['sample_a']]; rb = self.m[row['sample_b']]
        da, _ = load_graph_from_rec(ra, use_relations=False)
        db, _ = load_graph_from_rec(rb, use_relations=False)
        y = float(row['label']); w = float(row.get('weight', 1.0))
        return da, db, y, w

def collate_pairs(batch):
    das = [b[0] for b in batch]; dbs=[b[1] for b in batch]
    import torch
    ya = torch.tensor([b[2] for b in batch], dtype=torch.float32)
    wa = torch.tensor([b[3] for b in batch], dtype=torch.float32)
    from torch_geometric.data import Batch
    Ba = Batch.from_data_list(das); Bb=Batch.from_data_list(dbs)
    return Ba, Bb, ya, wa
