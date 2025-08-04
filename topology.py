import numpy as np
from shapely.geometry import Point, LineString, Polygon

def line_straightness(line: LineString) -> float:
    if line.length == 0: return 1.0
    chord = Point(line.coords[0]).distance(Point(line.coords[-1]))
    return float(chord / max(line.length, 1e-9))

def line_curvature_index(line: LineString) -> float:
    s = max(1e-9, line.length)
    chord = Point(line.coords[0]).distance(Point(line.coords[-1]))
    return float(1 - chord / s)

def polygon_circularity(poly: Polygon) -> float:
    A = max(1e-9, poly.area); P = max(1e-9, poly.length)
    return float(4*np.pi*A/(P*P))

def polygon_rectangularity(poly: Polygon) -> float:
    try:
        mbr = poly.minimum_rotated_rectangle
        return float(poly.area / max(mbr.area, 1e-9))
    except Exception:
        return 0.0

def polygon_convexity(poly: Polygon) -> float:
    try:
        hull = poly.convex_hull
        return float(poly.area / max(hull.area, 1e-9))
    except Exception:
        return 0.0

def topology_edges(points, lines, districts, tol: float):
    edges = []
    for p_idx, p in points:
        for d_idx, poly in districts:
            if poly.is_empty: continue
            if poly.buffer(tol).contains(p):
                edges.append((p_idx, d_idx, "TOPO_in_district"))
            elif poly.buffer(tol).boundary.distance(p) <= tol:
                edges.append((p_idx, d_idx, "TOPO_touches_boundary"))
    for p_idx, p in points:
        for l_idx, line in lines:
            if line.is_empty: continue
            if line.buffer(tol).contains(p):
                edges.append((p_idx, l_idx, "TOPO_on_line"))
            try:
                c0 = Point(line.coords[0]); c1 = Point(line.coords[-1])
                if p.distance(c0) <= tol or p.distance(c1) <= tol:
                    edges.append((p_idx, l_idx, "TOPO_near_endpoint"))
            except Exception:
                pass
    return edges
