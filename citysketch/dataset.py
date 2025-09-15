import random, numpy as np, pandas as pd, geopandas as gpd
from torch_geometric.data import Batch
from .graphs import build_pyg_from_layers

def rotate_nodes(nodes_df, theta):
    R = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    XY = nodes_df[['x','y']].values
    XYr = XY @ R.T
    out = nodes_df.copy()
    out[['x','y']] = XYr
    return out

def jitter_nodes(nodes_df, sigma=0.02):
    XY = nodes_df[['x','y']].values
    XYj = XY + np.random.normal(scale=sigma, size=XY.shape)
    out = nodes_df.copy(); out[['x','y']] = XYj
    return out

def to_pyg(nodes_df, sectors, use_relations=False):
    pts = nodes_df[nodes_df['Type'].isin(['Landmark','Node'])].copy(); pts['geometry']=None
    ln  = nodes_df[nodes_df['Type'].isin(['Path','Edge'])].copy(); ln['geometry']=None
    pg  = nodes_df[nodes_df['Type'].isin(['District'])].copy(); pg['geometry']=None
    gdf_pts = gpd.GeoDataFrame(pts, geometry='geometry'); gdf_ln = gpd.GeoDataFrame(ln, geometry='geometry'); gdf_pg = gpd.GeoDataFrame(pg, geometry='geometry')
    d, _ = build_pyg_from_layers(gdf_pts, gdf_ln, gdf_pg, relations_df=None, sectors=sectors, k=5, use_relations=use_relations, topo_tol_units=0.5)
    return d

class ContrastiveDataset:
    def __init__(self, recs):
        self.recs = recs
    def __len__(self): return len(self.recs)
    def __getitem__(self, idx):
        rec = self.recs[idx]
        nodes = pd.DataFrame(rec['nodes'])
        # two views: random rotation + small jitter
        th1 = np.random.uniform(0, 2*np.pi); th2 = np.random.uniform(0, 2*np.pi)
        n1 = jitter_nodes(rotate_nodes(nodes, th1), sigma=0.02)
        n2 = jitter_nodes(rotate_nodes(nodes, th2), sigma=0.02)
        d1 = to_pyg(n1, rec['sectors'], use_relations=False)
        d2 = to_pyg(n2, rec['sectors'], use_relations=False)
        return d1, d2

def collate_contrastive(batch):
    d1 = [b[0] for b in batch]; d2=[b[1] for b in batch]
    B1 = Batch.from_data_list(d1); B2=Batch.from_data_list(d2)
    return B1, B2


def anisotropic_scale(nodes_df, sx=None, sy=None, range_scale=(0.8,1.2)):
    import numpy as np
    if sx is None or sy is None:
        sx = np.random.uniform(*range_scale); sy = np.random.uniform(*range_scale)
    XY = nodes_df[['x','y']].values.astype(float)
    S = np.array([[sx,0.0],[0.0,sy]], dtype=float)
    XY2 = XY @ S.T
    out = nodes_df.copy(); out[['x','y']] = XY2
    return out

def elastic_deform(nodes_df, sigma=0.01, alpha=0.02):
    import numpy as np
    XY = nodes_df[['x','y']].values.astype(float)
    N=len(XY); 
    if N==0: return nodes_df
    disp = np.zeros_like(XY)
    for _ in range(2):
        direction = np.random.randn(2); direction/= (np.linalg.norm(direction)+1e-6)
        phase = np.random.rand()
        disp += alpha * np.sin(2*np.pi*(XY@direction + phase))[:,None] * direction
    disp += np.random.randn(*XY.shape)*sigma
    XY2 = XY + disp
    out = nodes_df.copy(); out[['x','y']] = XY2
    return out
