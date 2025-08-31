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

def build_ssg_and_signatures(df, sectors=8, k=5, per_sector_K=1, adaptive=True, radii=(0.75,1.0,1.5)):
    import numpy as np
    from sklearn.neighbors import NearestNeighbors
    X = df[["x","y"]].values.astype(float); n=len(X)
    if n==0:
        return [], np.zeros((0, sectors*len(radii))), X
    kn = min(max(8, k+3), n)
    nbrs = NearestNeighbors(n_neighbors=kn).fit(X)
    dists, idxs = nbrs.kneighbors(X)
    global_med = np.median(dists[:,2:]) if dists.shape[1] >= 3 else np.median(dists[:,1:])
    edges=set()
    def add(i,j):
        if i==j: return
        key=(int(i),int(j))
        if key not in edges: edges.add(key)
    for i in range(n):
        local_med = np.median(dists[i,2:]) if dists.shape[1] >= 3 else np.median(dists[i,1:])
        if not np.isfinite(local_med) or local_med<=1e-8: local_med = global_med if np.isfinite(global_med) else 1.0
        k_i = int(np.clip(np.round(k*(global_med/max(1e-8, local_med))), 3, 8)) if adaptive else k
        for j in idxs[i,1:k_i+1]: add(i,int(j))
        # per-sector K=1
        sec_best = {}
        for j in idxs[i,1:]:
            v = X[j]-X[i]; th=np.arctan2(v[1],v[0])
            z = df.iloc[i]['dir']; 
            if z is None or (isinstance(z,float) and np.isnan(z)): z = th
            rth=(th-z)%(2*np.pi); sec=int(np.floor((rth/(2*np.pi))*sectors))%sectors
            dd=float(np.linalg.norm(v))
            if sec not in sec_best or dd < sec_best[sec][0]: sec_best[sec]=(dd,int(j))
        for sec,(dd,jj) in sec_best.items(): add(i,jj)
    edges=list(edges)
    # multi-scale hist
    S=sectors; R=len(radii); sig=np.zeros((n,S*R), dtype=float)
    for i in range(n):
        # neighbors of i
        nbrs_i = [j for (a,b) in edges for j in ([b] if a==i else ([a] if b==i else []))]
        if not nbrs_i: continue
        xi = X[i]; z = df.iloc[i]['dir']
        if z is None or (isinstance(z,float) and np.isnan(z)):
            d2=((X-xi)**2).sum(axis=1); d2[i]=1e9; j0=int(np.argmin(d2)); z=np.arctan2(X[j0,1]-xi[1], X[j0,0]-xi[0])
        for j in nbrs_i:
            v = X[j]-xi; dd=np.linalg.norm(v)
            if dd<1e-8: continue
            th=np.arctan2(v[1],v[0]); rth=(th-z)%(2*np.pi)
            sec = int(np.floor((rth/(2*np.pi))*S))%S
            for ri,rad in enumerate(radii):
                gate=np.exp(-0.5*((dd/(rad+1e-8)-1.0)/0.5)**2)
                w = gate*(1.0/(dd+1e-3))
                sig[i, ri*S + sec] += w
        s = sig[i].sum()
        if s>0: sig[i]/=s
    return edges, sig, X

