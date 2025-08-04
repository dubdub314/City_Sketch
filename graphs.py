import numpy as np, torch
from torch_geometric.data import Data
from shapely.geometry import Point, LineString, Polygon
import pandas as pd
from .features import build_node_features, node_type_onehot, relation_degree_features
from .constants import EDGE_CAT
from .ssg import build_ssg_and_signatures, representative_point_and_dir
from .topology import topology_edges, line_straightness, line_curvature_index, polygon_circularity, polygon_rectangularity, polygon_convexity

def edge_type_index(tag: str) -> int:
    from .constants import EDGE_CAT
    return EDGE_CAT.get(tag, 0)

def edge_features_from_edges(nodes_df, edges, sectors=8, tag="SSG"):
    X = nodes_df[["x","y"]].values
    n = len(nodes_df)
    if n<=1 or len(edges)==0:
        return np.zeros((0, 3+sectors+len(EDGE_CAT))), np.zeros((2,0), dtype=int)
    edges = np.array(edges, dtype=int)
    edges_ud = np.vstack([edges, edges[:, ::-1]])
    src = edges_ud[:,0]; dst = edges_ud[:,1]
    vec = X[dst] - X[src]
    d = np.linalg.norm(vec, axis=1) + 1e-6
    r_norm = d
    zero = nodes_df["dir"].values
    zero = np.where(np.isnan(zero), 0.0, zero)
    ang = np.arctan2(vec[:,1], vec[:,0])
    rel = (ang - zero) % (2*np.pi)
    cos_t = np.cos(rel); sin_t = np.sin(rel)
    bins = np.linspace(0, 2*np.pi, sectors+1)
    sector_idx = np.clip(np.digitize(rel, bins) - 1, 0, sectors-1)
    sector_oh = np.eye(sectors)[sector_idx]
    etype_oh = np.zeros((len(EDGE_CAT),), dtype=float)
    etype_oh[edge_type_index(tag)] = 1.0
    etype_oh = np.tile(etype_oh, (len(edges_ud),1))
    edge_attr = np.concatenate([r_norm[:,None], cos_t[:,None], sin_t[:,None], sector_oh, etype_oh], axis=1)
    edge_index = edges_ud.T.astype(np.int64)
    return edge_attr, edge_index

def build_shape_descriptors(nodes_df, geom_records):
    n = len(nodes_df)
    D = np.zeros((n, 3), dtype=float)
    for idx, geom, t in geom_records:
        if t in ("Path","Edge") and isinstance(geom, LineString):
            D[idx,0] = line_straightness(geom)
            D[idx,1] = line_curvature_index(geom)
        elif t=="District" and isinstance(geom, Polygon):
            D[idx,0] = polygon_circularity(geom)
            D[idx,1] = polygon_rectangularity(geom)
            D[idx,2] = polygon_convexity(geom)
    return D

def add_relation_edges(edge_attr, edge_index, nodes_df, mapped_relations, sectors=8, w_rel=0.5):
    if not mapped_relations:
        return edge_attr, edge_index
    X = nodes_df[["x","y"]].values
    zero = nodes_df["dir"].values
    rel_types = {"Gaze":"REL_Gaze","Connection":"REL_Connection","Navigation":"REL_Navigation","Fusion":"REL_Fusion","Abruptness":"REL_Abruptness","Penetration":"REL_Penetration"}
    new_edges = []; new_attrs = []
    bins = np.linspace(0, 2*np.pi, sectors+1)
    for r in mapped_relations:
        s = r["src"]; t = r["dst"]
        if s<0 or t<0 or s>=len(nodes_df) or t>=len(nodes_df): 
            continue
        vec = X[t] - X[s]
        d = np.linalg.norm(vec) + 1e-6
        ang = np.arctan2(vec[1], vec[0])
        z = 0.0 if np.isnan(zero[s]) else zero[s]
        relang = (ang - z) % (2*np.pi)
        cos_t = np.cos(relang); sin_t = np.sin(relang)
        sec = np.clip(np.digitize([relang], bins)[0]-1, 0, sectors-1)
        sector_oh = np.eye(sectors)[sec]
        etype_oh = np.zeros((len(EDGE_CAT),), dtype=float)
        etype_oh[edge_type_index(rel_types.get(r.get("rel_type"), "REL_Connection"))] = 1.0 * w_rel
        attr = np.concatenate([[d, cos_t, sin_t], sector_oh, etype_oh], axis=0)
        new_edges.append([s,t]); new_attrs.append(attr)
    if new_edges:
        ei2 = np.array(new_edges, dtype=int).T
        ea2 = np.array(new_attrs, dtype=float)
        edge_index = np.concatenate([edge_index, ei2], axis=1)
        edge_attr = np.concatenate([edge_attr, ea2], axis=0)
    return edge_attr, edge_index

def build_pyg_from_layers(points_gdf, lines_gdf, polys_gdf, relations_df=None, sectors=8, k=5, use_relations=True, topo_tol_units=0.5):
    rows = []
    geom_records = []
    def push(gdf, declared_type):
        nonlocal rows, geom_records
        if gdf is None or gdf.empty: return
        for i, row in gdf.iterrows():
            geom = row.geometry
            name = str(row.get("Name","")) if row.get("Name") is not None else ""
            typ = str(row.get("Type", declared_type))
            pt, ang = representative_point_and_dir(geom)
            if pt is None: continue
            node_idx = len(rows)
            rows.append({"Name": name, "Type": typ, "x": pt[0], "y": pt[1], "dir": ang})
            geom_records.append((node_idx, geom, typ))
    push(points_gdf, "Landmark")
    push(lines_gdf, "Path")
    push(polys_gdf, "District")

    import pandas as pd
    nodes = pd.DataFrame(rows)
    if nodes.empty:
        import torch
        return Data(x=torch.zeros((0,1))), {"nodes": pd.DataFrame(), "scale":1.0, "relations_mapped":[], "relations_unresolved":[]}

    from .ssg import normalize_coords, build_ssg_and_signatures
    nodes_norm, scale = normalize_coords(nodes)
    ssg_edges, ang_sig, _ = build_ssg_and_signatures(nodes_norm, k=k, sectors=sectors)
    rel_mapped = []; unresolved = []
    if use_relations and relations_df is not None:
        from .relations import map_relations_by_name
        rel_mapped, unresolved = map_relations_by_name(relations_df, nodes_norm.rename(columns={"x":"x","y":"y","Name":"Name"}))
    rel_deg = None
    if use_relations:
        from .features import relation_degree_features
        rel_deg = relation_degree_features(rel_mapped, len(nodes_norm))
    shape_desc = build_shape_descriptors(nodes_norm, geom_records)
    types_oh = node_type_onehot(nodes_norm["Type"].tolist())
    from .features import build_node_features
    X = build_node_features(ang_sig, rel_deg, types_oh, shape_desc, w_rel=0.2 if use_relations else 0.0, w_type=0.2, w_shape=0.3)

    eattr, eidx = edge_features_from_edges(nodes_norm, ssg_edges, sectors=sectors, tag="SSG")
    # topology edges -- here we reuse original geoms (not normalized), but tol is in normalized units, used geometrically as buffer magnitude is small and invariant at scale here for qualitative checks
    pts = [(i, Point(r["x"], r["y"])) for i,r in nodes_norm.iterrows() if r["Type"] in ("Landmark","Node")]
    lines = []; polys = []
    for idx, (node_idx, geom, t) in enumerate(geom_records):
        if t in ("Path","Edge") and isinstance(geom, LineString):
            lines.append((node_idx, geom))
        elif t=="District" and isinstance(geom, Polygon):
            polys.append((node_idx, geom))
    topo = topology_edges(pts, lines, polys, tol=topo_tol_units)
    if topo:
        topo_edges = [(s,t) for (s,t,tag) in topo]
        topo_attr_list = []
        Xn = nodes_norm[["x","y"]].values
        zero = nodes_norm["dir"].values
        bins = np.linspace(0, 2*np.pi, sectors+1)
        for s,t,tag in topo:
            vec = Xn[t] - Xn[s]
            d = np.linalg.norm(vec)+1e-6
            ang = np.arctan2(vec[1], vec[0])
            z = 0.0 if np.isnan(zero[s]) else zero[s]
            rel = (ang - z) % (2*np.pi)
            cos_t = np.cos(rel); sin_t = np.sin(rel)
            sec = np.clip(np.digitize([rel], bins)[0]-1, 0, sectors-1)
            sector_oh = np.eye(sectors)[sec]
            etype_oh = np.zeros((len(EDGE_CAT),), dtype=float); etype_oh[edge_type_index(f"{tag}")] = 1.0 * 0.7
            attr = np.concatenate([[d, cos_t, sin_t], sector_oh, etype_oh], axis=0)
            topo_attr_list.append(attr)
        ea2 = np.array(topo_attr_list, dtype=float)
        ei2 = np.array(topo_edges, dtype=int).T
        eidx = np.concatenate([eidx, ei2], axis=1)
        eattr = np.concatenate([eattr, ea2], axis=0)
    if use_relations and rel_mapped:
        eattr, eidx = add_relation_edges(eattr, eidx, nodes_norm, rel_mapped, sectors=sectors, w_rel=0.5)

    import torch
    data = Data(
        x=torch.tensor(X, dtype=torch.float32),
        edge_index=torch.tensor(eidx, dtype=torch.long),
        edge_attr=torch.tensor(eattr, dtype=torch.float32),
        y=torch.tensor([0], dtype=torch.long)
    )
    data.num_nodes = X.shape[0]
    meta = {"nodes": nodes_norm, "scale": scale, "relations_mapped": rel_mapped, "relations_unresolved": unresolved}
    return data, meta
