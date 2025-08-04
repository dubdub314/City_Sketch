import numpy as np
from shapely.geometry import Point, LineString, Polygon

def representative_point_and_dir(geom):
    if geom.is_empty:
        return None, None
    if isinstance(geom, Point):
        return (geom.x, geom.y), None
    if isinstance(geom, LineString):
        try:
            p1 = geom.interpolate(geom.length*0.25)
            p2 = geom.interpolate(geom.length*0.75)
            ang = np.arctan2(p2.y - p1.y, p2.x - p1.x)
        except Exception:
            ang = None
        mid = geom.interpolate(geom.length*0.5)
        return (mid.x, mid.y), ang
    if isinstance(geom, Polygon):
        rp = geom.representative_point()
        return (rp.x, rp.y), None
    c = geom.centroid
    return (c.x, c.y), None

def normalize_coords(df):
    X = df[["x","y"]].values.astype(float)
    X = X - X.mean(axis=0, keepdims=True)
    if len(X) > 1:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=min(5, len(X))).fit(X)
        dists, _ = nn.kneighbors(X)
        med = np.median(dists[:,1:].ravel())
        scale = med if med>0 else 1.0
    else:
        scale = 1.0
    Xn = X/scale
    out = df.copy(); out[["x","y"]] = Xn
    return out, scale

def build_ssg_and_signatures(df, k=5, sectors=8):
    X = df[["x","y"]].values
    n = len(df)
    if n==0:
        return np.zeros((0,2),dtype=int), np.zeros((0,sectors)), X
    from sklearn.neighbors import NearestNeighbors
    k_eff = min(k+1, n)
    nn = NearestNeighbors(n_neighbors=k_eff).fit(X)
    dists, idxs = nn.kneighbors(X)
    edges = set()
    for i in range(n):
        for j in idxs[i,1:]:
            a,b = (i, int(j)) if i<int(j) else (int(j), i)
            edges.add((a,b))
    sector_edges = set()
    for i in range(n):
        vecs = X - X[i]
        angs = np.arctan2(vecs[:,1], vecs[:,0])
        d2 = (vecs**2).sum(axis=1)
        zero = df.iloc[i]["dir"]
        if zero is None or (isinstance(zero,float) and np.isnan(zero)):
            order = np.argsort(d2 + (np.arange(n)==i)*1e9)
            j0 = order[0] if order.size>0 else i
            zero = angs[j0]
        rel = (angs - zero) % (2*np.pi)
        bins = np.linspace(0, 2*np.pi, sectors+1)
        for s in range(sectors):
            mask = (rel>=bins[s]) & (rel<bins[s+1])
            cand = np.where(mask)[0]
            cand = cand[cand!=i]
            if cand.size>0:
                j = cand[np.argmin(d2[cand])]
                a,b = (i, int(j)) if i<int(j) else (int(j), i)
                sector_edges.add((a,b))
    edges |= sector_edges
    edges = np.array(sorted(list(edges)), dtype=int) if edges else np.zeros((0,2),dtype=int)
    sig = np.zeros((n, sectors), dtype=float)
    for i in range(n):
        nbrs = [j for (a,b) in edges for j in ([b] if a==i else ([a] if b==i else []))]
        if not nbrs: continue
        zero = df.iloc[i]["dir"]
        if zero is None or (isinstance(zero,float) and np.isnan(zero)):
            d2 = ((X - X[i])**2).sum(axis=1); d2[i]=1e9
            j0 = int(np.min(np.where(d2==d2.min())[0])) if np.any(d2<1e9) else 0
            zero = np.arctan2(X[j0,1]-X[i,1], X[j0,0]-X[i,0])
        vecs = X[nbrs] - X[i]
        angs = (np.arctan2(vecs[:,1], vecs[:,0]) - zero) % (2*np.pi)
        d = (vecs**2).sum(axis=1)**0.5; w = 1.0/(d+1e-6)
        hist, _ = np.histogram(angs, bins=sectors, range=(0,2*np.pi), weights=w)
        if hist.sum()>0: hist = hist/hist.sum()
        sig[i,:] = hist
    return edges, sig, X
