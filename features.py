import numpy as np
from .constants import REL_TYPES, TYPE_VALUES, EDGE_CAT

def relation_degree_features(mapped_relations, n_nodes):
    idx = {t:i for i,t in enumerate(REL_TYPES)}
    R = np.zeros((n_nodes, len(REL_TYPES)), dtype=float)
    for r in mapped_relations:
        t = r.get("rel_type")
        if t not in idx: 
            continue
        k = idx[t]
        R[r["src"], k] += 1.0
        R[r["dst"], k] += 1.0
    row_sums = R.sum(axis=1, keepdims=True)+1e-6
    return R/row_sums

def node_type_onehot(types):
    idx = {t:i for i,t in enumerate(TYPE_VALUES)}
    T = np.zeros((len(types), len(TYPE_VALUES)), dtype=float)
    for i, t in enumerate(types):
        if t in idx:
            T[i, idx[t]] = 1.0
    return T

def build_node_features(angular_sig, rel_deg, type_onehot, shape_desc=None, w_rel=0.2, w_type=0.2, w_shape=0.3):
    parts = [angular_sig]
    if rel_deg is not None:
        parts.append(w_rel*rel_deg)
    parts.append(w_type*type_onehot)
    if shape_desc is not None:
        parts.append(w_shape*shape_desc)
    return np.concatenate(parts, axis=1)

def edge_type_onehot(tag: str, num_edge_cats=11):
    oh = np.zeros((num_edge_cats,), dtype=float)
    from .constants import EDGE_CAT
    idx = EDGE_CAT.get(tag, 0)
    oh[idx] = 1.0
    return oh


def line_shape_descriptors(coords):
    import numpy as np
    if len(coords)<2: return np.array([1.0,0.0])
    L = np.sum(np.linalg.norm(coords[1:]-coords[:-1], axis=1))
    chord = np.linalg.norm(coords[-1]-coords[0])
    straight = chord / max(L,1e-6); curv = 1.0 - straight
    return np.array([straight, curv])

def polygon_shape_descriptors(coords):
    import shapely.geometry as sgeom, numpy as np
    poly = sgeom.Polygon(coords)
    if not poly.is_valid: poly = poly.buffer(0)
    if poly.area<=1e-12 or poly.length<=1e-12: return np.array([0.0,0.0,0.0,0.0])
    A=poly.area; P=poly.length
    circ = 4.0*np.pi*A/(P*P)
    mrr = poly.minimum_rotated_rectangle
    rect = A / max(mrr.area,1e-12)
    conv = A / max(poly.convex_hull.area,1e-12)
    tf_fd = turning_function_fd(poly, k=4)
    return np.array([circ, rect, conv, tf_fd])

def turning_function_fd(poly, k=4):
    import numpy as np
    coords = np.asarray(poly.exterior.coords, dtype=float)
    if len(coords)<4: return 0.0
    v = np.diff(coords, axis=0)
    ang = np.arctan2(v[:,1], v[:,0])
    dang = np.diff(ang, prepend=ang[0])
    dang = (dang + np.pi)%(2*np.pi) - np.pi
    fft = np.fft.rfft(dang - dang.mean())
    mags = np.abs(fft[1:k+1])
    if mags.size==0: return 0.0
    return float(np.mean(mags) / (np.abs(fft).sum()+1e-6))
