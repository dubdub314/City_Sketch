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
                from shapely.geometry import Point as Pt
                c0 = Pt(line.coords[0]); c1 = Pt(line.coords[-1])
                if p.distance(c0) <= tol or p.distance(c1) <= tol:
                    edges.append((p_idx, l_idx, "TOPO_near_endpoint"))
            except Exception:
                pass
    return edges


def _soft_w(d, tau=0.5):
    import numpy as np
    return float(np.exp(-d / max(1e-6, tau)))

def soft_topology_edges(points, lines, polys, tol_units=0.5):
    edges = []
    for p_idx, p in points:
        for d_idx, poly in polys:
            d = 0.0 if poly.contains(p) else p.distance(poly)
            w = _soft_w(d, tau=tol_units)
            if w>1e-4: edges.append((p_idx, d_idx, "TOPO_in_district", w))
            try:
                bdist = poly.boundary.distance(p)
                w2 = _soft_w(bdist, tau=tol_units)
                if w2>1e-4:
                    edges.append((p_idx, d_idx, "TOPO_touches_boundary", w2))
            except Exception:
                pass
    for p_idx, p in points:
        for l_idx, line in lines:
            d = line.distance(p)
            w = _soft_w(d, tau=tol_units)
            if w>1e-4: edges.append((p_idx, l_idx, "TOPO_on_line", w))
            try:
                from shapely.geometry import Point as Pt
                c0 = Pt(line.coords[0]); c1 = Pt(line.coords[-1])
                d0 = p.distance(c0); d1 = p.distance(c1)
                w0 = _soft_w(d0, tau=tol_units); w1 = _soft_w(d1, tau=tol_units)
                if w0>1e-4: edges.append((p_idx, l_idx, "TOPO_near_endpoint", 0.5*w0))
                if w1>1e-4: edges.append((p_idx, l_idx, "TOPO_near_endpoint", 0.5*w1))
            except Exception:
                pass
    return edges
