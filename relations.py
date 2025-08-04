from collections import defaultdict
from .constants import REL_TYPES

def map_relations_by_name(rel_df, nodes_df):
    if rel_df is None or rel_df.empty:
        return [], []
    cols = [c.lower() for c in rel_df.columns]
    try:
        si = cols.index("src_name"); di = cols.index("dst_name"); ti = cols.index("rel_type")
    except ValueError:
        return [], []
    src_col = rel_df.columns[si]; dst_col = rel_df.columns[di]; type_col = rel_df.columns[ti]
    name_to_idx = defaultdict(list)
    for i, nm in enumerate(nodes_df["Name"].fillna("").astype(str).tolist()):
        nm = nm.strip()
        if nm:
            name_to_idx[nm.lower()].append(i)
    mapped, unresolved = [], []
    for _, row in rel_df.iterrows():
        sname = str(row[src_col]).strip() if row.get(src_col) is not None else ""
        dname = str(row[dst_col]).strip() if row.get(dst_col) is not None else ""
        rtype = str(row[type_col]).strip() if row.get(type_col) is not None else ""
        if not sname or not dname: continue
        si_list = name_to_idx.get(sname.lower(), [])
        di_list = name_to_idx.get(dname.lower(), [])
        if si_list and di_list and rtype in REL_TYPES:
            mapped.append({"src": si_list[0], "dst": di_list[0], "rel_type": rtype})
        else:
            unresolved.append({"src_name": sname, "dst_name": dname, "rel_type": rtype})
    return mapped, unresolved
