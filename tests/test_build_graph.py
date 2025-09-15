import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
from citysketch.graphs import build_pyg_from_layers

def test_build_minimal():
    pts = gpd.GeoDataFrame({'Name':['A'],'Type':['Landmark']}, geometry=[Point(0,0)], crs="EPSG:4326")
    lns = gpd.GeoDataFrame({'Name':['L1'],'Type':['Path']}, geometry=[LineString([(0,0),(1,0)])], crs="EPSG:4326")
    polys = gpd.GeoDataFrame({'Name':['D1'],'Type':['District']}, geometry=[Polygon([(0,0),(0,1),(1,1),(1,0)])], crs="EPSG:4326")
    data = build_pyg_from_layers(dict(points=pts, lines=lns, polygons=polys), use_relations=False, sectors=8, k=5, topo_tol=0.5)
    assert data.x.shape[0] >= 3
    assert data.edge_index.size(1) > 0
