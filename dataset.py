import json, random, numpy as np, pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset
from .graphs import build_pyg_from_layers
from .ssg import build_ssg_and_signatures

def rotate_coords(nodes_df, theta):
    c = np.cos(theta); s = np.sin(theta)
    XY = nodes_df[['x','y']].values.copy()
    R = np.array([[c,-s],[s,c]]); XY = XY @ R.T
    out = nodes_df.copy(); out['x']=XY[:,0]; out['y']=XY[:,1]
    d = out['dir'].values.copy()
    d = np.where(np.isnan(d), d, (d+theta) % (2*np.pi))
    out['dir'] = d
    return out

def jitter_nodes(nodes_df, sigma=0.02):
    XY = nodes_df[['x','y']].values.copy()
    XY += np.random.normal(scale=sigma, size=XY.shape)
    out = nodes_df.copy(); out[['x','y']] = XY
    return out

class GraphPairDataset(Dataset):
    def __init__(self, manifest_paths, use_relations=True, sectors=8):
        self.samples = [json.loads(Path(p).read_text(encoding='utf-8')) for p in manifest_paths]
        self.sectors = sectors; self.use_relations = use_relations

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        rec = self.samples[idx]
        nodes = pd.DataFrame(rec['nodes'])
        th1 = random.uniform(-np.pi, np.pi); th2 = random.uniform(-np.pi, np.pi)
        n1 = rotate_coords(nodes, th1); n2 = rotate_coords(nodes, th2)
        n1 = jitter_nodes(n1, 0.01); n2 = jitter_nodes(n2, 0.01)
        pts1 = n1[n1['Type'].isin(['Landmark','Node'])].copy(); pts1['geometry']=None
        ln1 = n1[n1['Type'].isin(['Path','Edge'])].copy(); ln1['geometry']=None
        pg1 = n1[n1['Type'].isin(['District'])].copy(); pg1['geometry']=None
        import geopandas as gpd
        gdf_pts1 = gpd.GeoDataFrame(pts1, geometry='geometry'); gdf_ln1=gpd.GeoDataFrame(ln1, geometry='geometry'); gdf_pg1=gpd.GeoDataFrame(pg1, geometry='geometry')
        d1, _ = build_pyg_from_layers(gdf_pts1, gdf_ln1, gdf_pg1, relations_df=None, sectors=self.sectors, k=5, use_relations=self.use_relations, topo_tol_units=0.5)
        pts2 = n2[n2['Type'].isin(['Landmark','Node'])].copy(); pts2['geometry']=None
        ln2 = n2[n2['Type'].isin(['Path','Edge'])].copy(); ln2['geometry']=None
        pg2 = n2[n2['Type'].isin(['District'])].copy(); pg2['geometry']=None
        gdf_pts2 = gpd.GeoDataFrame(pts2, geometry='geometry'); gdf_ln2=gpd.GeoDataFrame(ln2, geometry='geometry'); gdf_pg2=gpd.GeoDataFrame(pg2, geometry='geometry')
        d2, _ = build_pyg_from_layers(gdf_pts2, gdf_ln2, gdf_pg2, relations_df=None, sectors=self.sectors, k=5, use_relations=self.use_relations, topo_tol_units=0.5)
        return d1, d2
