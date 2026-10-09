"""
Dynamic Adjacency Matrix Construction and Sparsification
"""

import numpy as np

# Standard 2D 10-20 Normalized Scalp Coordinates
ELECTRODE_POSITIONS_2D = {
    "Fp1": (-0.35, 0.85), "Fp2": (0.35, 0.85),
    "F7":  (-0.80, 0.50), "F3":  (-0.40, 0.50), "Fz":  (0.00, 0.50), "F4":  (0.40, 0.50), "F8":  (0.80, 0.50),
    "T7":  (-0.90, 0.00), "C3":  (-0.45, 0.00), "Cz":  (0.00, 0.00), "C4":  (0.45, 0.00), "T8":  (0.90, 0.00),
    "P7":  (-0.80, -0.50), "P3": (-0.40, -0.50), "Pz": (0.00, -0.50), "P4": (0.40, -0.50), "P8": (0.80, -0.50),
    "O1":  (-0.35, -0.85), "O2": (0.35, -0.85)
}

def get_spatial_prior_matrix(channel_names=None) -> np.ndarray:
    """
    Construct anatomical spatial distance prior adjacency matrix.
    """
    if channel_names is None:
        channel_names = list(ELECTRODE_POSITIONS_2D.keys())
        
    n = len(channel_names)
    coords = np.array([ELECTRODE_POSITIONS_2D.get(ch, (0.0, 0.0)) for ch in channel_names])
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    
    # Exponential spatial decay
    prior = np.exp(-dist / 0.8)
    return prior.astype(np.float32)

def sparsify_adjacency(plv_matrix: np.ndarray,
                       threshold: float = 0.25,
                       top_k: int = None,
                       use_spatial_prior: bool = True,
                       spatial_weight: float = 0.3) -> np.ndarray:
    """
    Transform raw PLV connectivity into dynamic graph adjacency A_t.
    
    Parameters
    ----------
    plv_matrix : np.ndarray
        Shape (N, N)
    threshold : float
        Minimum PLV threshold to form an edge.
    top_k : int, optional
        If specified, keeps top-k strongest edges per node.
    use_spatial_prior : bool
        Whether to blend dynamic PLV with 10-20 anatomical distance.
    spatial_weight : float
        Blending ratio for spatial prior (e.g. 0.3 spatial + 0.7 PLV).
    """
    N = plv_matrix.shape[0]
    
    if use_spatial_prior and N == 19:
        spatial_prior = get_spatial_prior_matrix()
        adj = (1.0 - spatial_weight) * plv_matrix + spatial_weight * spatial_prior
    else:
        adj = plv_matrix.copy()
        
    if top_k is not None and top_k < N:
        sparse_adj = np.zeros_like(adj)
        for i in range(N):
            # Find indices of top-k largest weights (excluding self)
            row = adj[i, :].copy()
            row[i] = -1.0 # Ignore self for top-k selection
            top_indices = np.argsort(row)[-top_k:]
            sparse_adj[i, top_indices] = adj[i, top_indices]
            sparse_adj[top_indices, i] = adj[top_indices, i] # Symmetric
        adj = sparse_adj
    else:
        adj = np.where(adj >= threshold, adj, 0.0)
        
    np.fill_diagonal(adj, 1.0) # Self-loops for GAT node features
    return adj.astype(np.float32)
