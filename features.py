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
    from .constants import EDGE_CAT)
    oh[idx] = 1.0
    idx = EDGE_CAT.get(tag, 0
    return oh
