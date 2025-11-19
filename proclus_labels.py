"""
PROCLUS labels-only implementation (no O(n*k) C-lists)
Memory-safe for large k by using label vectors instead of cluster membership lists.
"""
import torch


@torch.no_grad()
def assign_points_labels_tiled(X, M, D_mask, tile_n=131072):
    """
    Assign each point to nearest medoid using projected L1 distance.
    Returns: labels [n] int64, C_sizes [k] int32
    """
    device = X.device
    n, d = X.shape
    k = M.numel()
    
    labels = torch.full((n,), -1, device=device, dtype=torch.long)
    C_sizes = torch.zeros((k,), device=device, dtype=torch.int32)
    
    Xm = X.index_select(0, M)  # [k, d]
    
    for p0 in range(0, n, tile_n):
        p1 = min(n, p0 + tile_n)
        Xp = X[p0:p1]  # [tile, d]
        
        # Compute projected L1 distances for this tile
        dists = torch.zeros((p1 - p0, k), device=device, dtype=X.dtype)
        for i in range(k):
            mask = D_mask[i]  # [d] bool
            if mask.any():
                diff = (Xp[:, mask] - Xm[i, mask]).abs()
                dists[:, i] = diff.mean(dim=1)
            else:
                dists[:, i] = float('inf')
        
        # Assign to nearest
        best_idx = torch.argmin(dists, dim=1)
        labels[p0:p1] = best_idx
        
        # Update sizes
        for i in range(k):
            C_sizes[i] += (best_idx == i).sum()
    
    return labels, C_sizes


@torch.no_grad()
def evaluate_cost_from_labels_tiled(X, labels, D_mask, C_sizes, tile_n=131072):
    """
    Compute cost = sum of mean absolute deviations per cluster.
    Returns: cost (scalar), means [k, d]
    """
    device = X.device
    n, d = X.shape
    k = D_mask.shape[0]
    
    means = torch.zeros((k, d), device=device, dtype=X.dtype)
    cost = torch.tensor(0.0, device=device, dtype=X.dtype)
    
    # Compute means per cluster
    for i in range(k):
        sel = (labels == i)
        if sel.any():
            means[i] = X[sel].mean(dim=0)
    
    # Compute cost in tiles
    for p0 in range(0, n, tile_n):
        p1 = min(n, p0 + tile_n)
        Lc = labels[p0:p1]
        valid = Lc >= 0
        if not valid.any():
            continue
        
        for i in torch.unique(Lc[valid]).tolist():
            i = int(i)
            dims = D_mask[i]
            if not dims.any():
                continue
            
            idx = (Lc == i).nonzero(as_tuple=True)[0]
            Xq = X[p0:p1][idx][:, dims]
            mu = means[i, dims]
            mad = (Xq - mu).abs().mean()
            cost += mad * idx.numel()
    
    return cost, means


@torch.no_grad()
def update_best_labels_py(cost, best_cost, M, labels, C_sizes, min_deviation,
                          M_best, labels_best, C_sizes_best, bad):
    """
    Update best solution if cost improved; mark bad clusters.
    Returns: best_cost, M_best, labels_best, C_sizes_best, bad, improved
    """
    improved = False
    if cost < best_cost:
        best_cost = cost.clone()
        M_best = M.clone()
        labels_best = labels.clone()
        C_sizes_best = C_sizes.clone()
        improved = True
    
    # Mark bad clusters (too small)
    bad.zero_()
    threshold = int(min_deviation * labels.numel())
    bad[:] = C_sizes_best < threshold
    
    return best_cost, M_best, labels_best, C_sizes_best, bad, improved


@torch.no_grad()
def fit_proclus_labels_py(X, k, l, a, b, min_deviation, termination_rounds, seed=123, tile_n=131072):
    """
    Memory-safe PROCLUS FIT using labels only (no O(n*k) arrays).
    
    Args:
        X: [n, d] float32 cuda tensor
        k: number of clusters
        l: number of dimensions per cluster
        a, b: candidate pool parameters
        min_deviation: minimum cluster size fraction
        termination_rounds: number of iterations
        seed: random seed
        tile_n: tile size for streaming operations
    
    Returns:
        M_best: [k] int64 medoid indices
        D_mask: [k, d] bool projection mask
    """
    device = X.device
    n, d = X.shape
    torch.manual_seed(seed)
    
    # Candidate pool via farthest-point sampling
    Bk = min(n, max(b * k, k + 1))
    cand = torch.randperm(n, device=device)[:Bk]
    
    M = torch.empty((k,), device=device, dtype=torch.long)
    M[0] = cand[0]
    d2 = torch.full((Bk,), float("inf"), device=device, dtype=X.dtype)
    
    for t in range(1, k):
        m_prev = X[M[t-1]]
        diff = X[cand] - m_prev
        d2 = torch.minimum(d2, (diff * diff).sum(dim=1))
        M[t] = cand[int(torch.argmax(d2))]
    
    # Initial D_mask (all dims)
    D_mask = torch.ones((k, d), device=device, dtype=torch.bool)
    
    # Best trackers
    best_cost = torch.tensor(float("inf"), device=device, dtype=X.dtype)
    M_best = M.clone()
    labels_best = torch.full((n,), -1, device=device, dtype=torch.long)
    C_sizes_best = torch.zeros((k,), device=device, dtype=torch.int32)
    bad = torch.zeros((k,), device=device, dtype=torch.bool)
    
    # Iterations
    for iteration in range(termination_rounds):
        # Assign
        labels, C_sizes = assign_points_labels_tiled(X, M, D_mask, tile_n=tile_n)
        
        # Evaluate
        cost, means = evaluate_cost_from_labels_tiled(X, labels, D_mask, C_sizes, tile_n=tile_n)
        
        # Update best
        best_cost, M_best, labels_best, C_sizes_best, bad, improved = update_best_labels_py(
            cost, best_cost, M, labels, C_sizes, min_deviation,
            M_best, labels_best, C_sizes_best, bad
        )
        
        # Recompute D_mask: select l dims with smallest MAD per cluster
        Z = torch.full((k, d), float("inf"), device=device, dtype=X.dtype)
        for i in range(k):
            sel = (labels == i)
            if sel.any():
                Xi = X[sel]
                mu = means[i]
                mad = (Xi - mu).abs().mean(dim=0)
                Z[i] = mad
        
        D_mask.zero_()
        l_eff = max(1, min(l, d))
        idx = torch.topk(-Z, k=l_eff, dim=1).indices
        D_mask.scatter_(1, idx, True)
        
        # Replace bad medoids
        if bad.any():
            used = torch.zeros(n, device=device, dtype=torch.bool)
            used[M] = True
            free = cand[~used[cand]]
            if free.numel() > 0:
                n_bad = int(bad.sum())
                repl = free[torch.randperm(free.numel(), device=device)[:n_bad]]
                M[bad] = repl
    
    # Final D_mask with best medoids
    labels, C_sizes = assign_points_labels_tiled(X, M_best, D_mask, tile_n=tile_n)
    Z = torch.full((k, d), float("inf"), device=device, dtype=X.dtype)
    for i in range(k):
        sel = (labels == i)
        if sel.any():
            mu = X[sel].mean(dim=0)
            mad = (X[sel] - mu).abs().mean(dim=0)
            Z[i] = mad
    
    D_mask.zero_()
    l_eff = max(1, min(l, d))
    idx = torch.topk(-Z, k=l_eff, dim=1).indices
    D_mask.scatter_(1, idx, True)
    
    return M_best, D_mask
