import numpy as np, ot
from scipy.optimize import linear_sum_assignment
import networkx as nx

def cosine_similarity_matrix(A, B, eps=1e-8):
    a = A / (np.linalg.norm(A, axis=1, keepdims=True) + eps)
    b = B / (np.linalg.norm(B, axis=1, keepdims=True) + eps)
    return a @ b.T

def hungarian_similarity(A, B):
    sims = cosine_similarity_matrix(A, B)
    costs = 1 - sims
    n, m = sims.shape
    size = max(n, m)
    C = np.ones((size,size))
    C[:n,:m] = costs
    ri, cj = linear_sum_assignment(C)
    pairs = [(i,j) for i,j in zip(ri,cj) if i<n and j<m]
    score = 1 - np.mean([costs[i,j] for i,j in pairs]) if pairs else 0.0
    coverage = len(pairs)/max(n,m) if max(n,m)>0 else 0.0
    return score, coverage, pairs, sims

def shortest_path_cost(adj):
    G = nx.Graph(); n = adj.shape[0]
    for i in range(n): G.add_node(i)
    xs, ys = np.where(adj>0)
    for i,j in zip(xs,ys):
        if i!=j: G.add_edge(i,j,weight=1.0/adj[i,j])
    D = np.zeros((n,n))
    for i in range(n):
        lengths = nx.single_source_dijkstra_path_length(G, i)
        for j, d in lengths.items():
            D[i,j] = d
    D = D / (np.median(D[D>0]) if np.any(D>0) else 1.0)
    return D

def fgw_similarity(node_feat_A, node_feat_B, adjA, adjB, alpha=0.8, reg=1e-3):
    sims = cosine_similarity_matrix(node_feat_A, node_feat_B)
    C_feat = 1 - (sims+1)/2.0
    DA = shortest_path_cost(adjA + 1e-6*np.eye(adjA.shape[0]))
    DB = shortest_path_cost(adjB + 1e-6*np.eye(adjB.shape[0]))
    p = ot.unif(DA.shape[0]); q = ot.unif(DB.shape[0])
    T = ot.gromov.fused_gromov_wasserstein(C_feat, DA, DB, p, q, 'square_loss', alpha=alpha, verbose=False, log=False, epsilon=reg)
    fgw_cost = np.sum(T * C_feat)
    sim = 1 - fgw_cost / (np.max(C_feat) + 1e-9)
    sim = float(np.clip(sim, 0.0, 1.0))
    return sim, T
