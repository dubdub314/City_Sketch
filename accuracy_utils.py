import numpy as np, pandas as pd
from typing import Dict

def match_by_name(dfA: pd.DataFrame, dfB: pd.DataFrame) -> Dict[int,int]:
    name_to_idx_B = {str(n).strip().lower(): i for i,n in enumerate(dfB['Name'].fillna('').astype(str)) if str(n).strip()}
    mapping = {}
    for i, n in enumerate(dfA['Name'].fillna('').astype(str)):
        nm = str(n).strip().lower()
        if nm and nm in name_to_idx_B:
            mapping[i] = name_to_idx_B[nm]
    return mapping

def type_consistency(dfA: pd.DataFrame, dfB: pd.DataFrame, mapping: Dict[int,int]) -> float:
    if not mapping: return 0.0
    ok = 0
    for i,j in mapping.items():
        ok += (dfA.iloc[i]['Type'] == dfB.iloc[j]['Type'])
    return ok / len(mapping)

def per_class_scores(dfA: pd.DataFrame, dfB: pd.DataFrame, mapping: Dict[int,int]):
    classes = {'points': ['Landmark','Node'], 'lines': ['Path','Edge'], 'polygons':['District']}
    out = {}
    for k, types in classes.items():
        matched_pairs = [(i, mapping[i]) for i in mapping if dfA.iloc[i]['Type'] in types and dfB.iloc[mapping[i]]['Type'] in types]
        if not matched_pairs:
            out[k] = None
        else:
            out[k] = len(matched_pairs) / max(sum(dfA['Type'].isin(types)), sum(dfB['Type'].isin(types)))
    return out
