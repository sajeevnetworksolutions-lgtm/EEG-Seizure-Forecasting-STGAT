# NeuroSync 4D: ST-GAT Clinical Neuro-Telemetry System (Release v2.4.2 - 2026-10-17)
"""
Dynamic Brain Graph Connectivity Construction and Phase Locking Value (PLV) Engine
"""

import numpy as np
import scipy.signal as signal
from src.config import NUM_CHANNELS, ELECTRODES_19, ELECTRODE_COORDS_2D

class DynamicGraphConstructor:
    """
    Constructs dynamic time-resolved spatial connectivity graph adjacency matrices A^t
    by computing pairwise Phase Locking Value (PLV) and Phase Lag Index (PLI)
    across all 19 electrode channels.
    """
    def __init__(self, num_channels=NUM_CHANNELS, plv_threshold=0.25, add_self_loops=True):
        self.num_channels = num_channels
        self.plv_threshold = plv_threshold
        self.add_self_loops = add_self_loops
        self.spatial_distance_matrix = self._compute_spatial_distance_matrix()

    def _compute_spatial_distance_matrix(self):
        """
        Compute normalized Euclidean distance matrix between standard 10-20 electrode positions.
        """
        coords = np.array([ELECTRODE_COORDS_2D[ch] for ch in ELECTRODES_19])
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))
        # Normalized distance-based prior adjacency (closer electrodes have higher base connectivity)
        spatial_prior = np.exp(-dist / 0.8)
        return spatial_prior.astype(np.float32)

    def compute_instantaneous_phase(self, eeg_epoch):
        """
        Compute instantaneous phase phi(t) for each channel using Hilbert transform.
        Args:
            eeg_epoch: (num_channels, epoch_samples)
        Returns:
            phase: (num_channels, epoch_samples) in [-pi, pi]
        """
        analytic_signal = signal.hilbert(eeg_epoch, axis=-1)
        phase = np.angle(analytic_signal)
        return phase

    def compute_plv_matrix(self, eeg_epoch):
        """
        Compute pairwise Phase Locking Value (PLV) matrix for a single EEG epoch.
        PLV_ij = (1 / N) * |sum_{t=1}^N exp(i * (phi_i(t) - phi_j(t)))|
        Args:
            eeg_epoch: (num_channels, epoch_samples)
        Returns:
            plv_matrix: (num_channels, num_channels) with values in [0, 1]
        """
        phase = self.compute_instantaneous_phase(eeg_epoch)
        num_ch = phase.shape[0]
        plv = np.zeros((num_ch, num_ch), dtype=np.float32)

        # Vectorized pairwise phase difference calculation
        # phase: (C, T) -> exp_phase: (C, T) complex
        exp_phase = np.exp(1j * phase) # shape (C, T)
        # Cross product across channels averaged over time
        # (C, T) @ (T, C) -> (C, C)
        plv = np.abs(np.dot(exp_phase, exp_phase.conj().T) / phase.shape[1]).astype(np.float32)
        
        # Ensure diagonal is 1.0 and clip bounds [0, 1]
        np.fill_diagonal(plv, 1.0)
        plv = np.clip(plv, 0.0, 1.0)
        return plv

    def compute_pli_matrix(self, eeg_epoch):
        """
        Compute pairwise Phase Lag Index (PLI) matrix to guard against volume conduction artifacts.
        PLI_ij = | (1/N) * sum_{t=1}^N sign(sin(phi_i(t) - phi_j(t))) |
        Args:
            eeg_epoch: (num_channels, epoch_samples)
        Returns:
            pli_matrix: (num_channels, num_channels)
        """
        phase = self.compute_instantaneous_phase(eeg_epoch)
        num_ch = phase.shape[0]
        pli = np.zeros((num_ch, num_ch), dtype=np.float32)
        
        for i in range(num_ch):
            for j in range(i, num_ch):
                if i == j:
                    pli[i, j] = 1.0
                else:
                    dphi = phase[i, :] - phase[j, :]
                    val = np.abs(np.mean(np.sign(np.sin(dphi))))
                    pli[i, j] = val
                    pli[j, i] = val
        return pli

    def construct_dynamic_adjacency(self, eeg_epoch, use_spatial_prior=True):
        """
        Construct the final dynamic adjacency matrix A^t for the ST-GAT model.
        Combines electrophysiological Phase Locking Value (PLV) with anatomical 10-20 spatial prior.
        Args:
            eeg_epoch: (num_channels, epoch_samples)
            use_spatial_prior: whether to fuse PLV with anatomical electrode distance
        Returns:
            adj: (num_channels, num_channels) float32 adjacency matrix
        """
        plv = self.compute_plv_matrix(eeg_epoch)
        
        if use_spatial_prior:
            # Dynamic graph topology fusion: 70% Phase synchrony + 30% Spatial anatomy
            adj = 0.70 * plv + 0.30 * self.spatial_distance_matrix
        else:
            adj = plv
            
        # Threshold weak noisy synchrony
        adj = np.where(adj >= self.plv_threshold, adj, 0.0)
        
        if self.add_self_loops:
            np.fill_diagonal(adj, 1.0)
            
        return adj.astype(np.float32)

    def construct_batch_adjacencies(self, batch_epochs, use_spatial_prior=True):
        """
        Construct adjacency matrices for a batch of epochs or sequences.
        Args:
            batch_epochs: np.ndarray of shape (B, num_channels, epoch_samples)
                          or (B, Seq, num_channels, epoch_samples)
        Returns:
            adj_matrices: (B, num_channels, num_channels) or (B, Seq, num_channels, num_channels)
        """
        if batch_epochs.ndim == 3:
            B, C, T = batch_epochs.shape
            adjs = np.zeros((B, C, C), dtype=np.float32)
            for b in range(B):
                adjs[b] = self.construct_dynamic_adjacency(batch_epochs[b], use_spatial_prior)
            return adjs
        elif batch_epochs.ndim == 4:
            B, Seq, C, T = batch_epochs.shape
            adjs = np.zeros((B, Seq, C, C), dtype=np.float32)
            for b in range(B):
                for s in range(Seq):
                    adjs[b, s] = self.construct_dynamic_adjacency(batch_epochs[b, s], use_spatial_prior)
            return adjs
        else:
            raise ValueError(f"Invalid input shape {batch_epochs.shape}")
