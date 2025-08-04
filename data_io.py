import re
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import pandas as pd
import geopandas as gpd

NAME_FIELD = "Name"
TYPE_FIELD = "Type"
RELATION_CAND = ["relations.xlsx","relations.csv","Relations.xlsx","Relations.csv"]

def list_samples(group_dir: str) -> List[Path]:
    root = Path(group_dir)
    out = []
    for p in sorted(root.glob("*")):
        if p.is_dir() and re.search(r"sample", p.name, re.IGNORECASE):
            out.append(p)
    if not out:
        for p in sorted(root.rglob("*")):
            if p.is_dir() and re.search(r"sample", p.name, re.IGNORECASE):
                out.append(p)
    return out

def read_relations(sample_dir: Path) -> Optional[pd.DataFrame]:
    for cand in RELATION_CAND:
        p = sample_dir/cand
        if p.exists():
            if p.suffix.lower()==".xlsx":
                return pd.read_excel(p, sheet_name=0)
            else:
                return pd.read_csv(p)
    for p in sample_dir.glob("Relations*.*"):
        try:
            if p.suffix.lower()==".xlsx":
                return pd.read_excel(p, sheet_name=0)
            elif p.suffix.lower()==".csv":
                return pd.read_csv(p)
        except Exception:
            continue
    return None

def read_shapefiles(sample_dir: Path) -> Dict[str, gpd.GeoDataFrame]:
    layers = {"points":[], "lines":[], "polygons":[]}
    for shp in sample_dir.rglob("*.shp"):
        try:
            gdf = gpd.read_file(shp)
        except Exception:
            continue
        if gdf.empty: 
            continue
        if NAME_FIELD not in gdf.columns or TYPE_FIELD not in gdf.columns:
            cols = {c.lower():c for c in gdf.columns}
            if "name" in cols: gdf = gdf.rename(columns={cols["name"]:NAME_FIELD})
            if "type" in cols: gdf = gdf.rename(columns={cols["type"]:TYPE_FIELD})
        geom_type = str(gdf.geom_type.iloc[0])
        if "Point" in geom_type:
            layers["points"].append(gdf)
        elif "Line" in geom_type:
            layers["lines"].append(gdf)
        elif "Polygon" in geom_type:
            layers["polygons"].append(gdf)
    merged = {}
    for k, lst in layers.items():
        if lst:
            merged[k] = pd.concat(lst, ignore_index=True)
        else:
            merged[k] = gpd.GeoDataFrame(columns=[NAME_FIELD, TYPE_FIELD, "geometry"], geometry="geometry", crs="EPSG:3857")
    return merged

def read_realmap(real_dir: str) -> Dict[str, gpd.GeoDataFrame]:
    return read_shapefiles(Path(real_dir))
