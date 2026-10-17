# NeuroSync 4D: ST-GAT Clinical Neuro-Telemetry System (Release v2.4.2 - 2026-10-17)
"""
Unit tests for Phase Locking Value (PLV) and Adjacency Matrix Construction
"""

import pytest
import numpy as np
from src.graph.plv import compute_plv, compute_band_plv
from src.graph.adjacency import sparsify_adjacency, get_spatial_prior_matrix

def test_plv_bounds_and_symmetry():
    # Generate 19-channel signal
    rng = np.random.RandomState(42)
    eeg = rng.randn(19, 500).astype(np.float32)
    
    plv = compute_plv(eeg, zero_diagonal=False)
    assert plv.shape == (19, 19)
    assert np.all(plv >= 0.0) and np.all(plv <= 1.0)
    assert np.allclose(plv, plv.T, atol=1e-5), "PLV must be symmetric"
    assert np.allclose(np.diag(plv), 1.0), "Self-synchrony diagonal must be 1.0"

def test_sparsify_adjacency():
    plv = np.random.uniform(0.1, 0.9, (19, 19)).astype(np.float32)
    plv = (plv + plv.T) / 2.0
    np.fill_diagonal(plv, 1.0)
    
    adj = sparsify_adjacency(plv, threshold=0.4, use_spatial_prior=True)
    assert adj.shape == (19, 19)
    assert np.all(adj >= 0.0)
    assert np.allclose(np.diag(adj), 1.0)
