# NeuroSync 4D: ST-GAT Clinical Neuro-Telemetry System (Release v2.4.2 - 2026-10-17)
"""
Phase Locking Value (PLV) Functional Connectivity Calculation
"""

import numpy as np
from scipy.signal import hilbert, butter, filtfilt

def compute_plv(eeg_window: np.ndarray, zero_diagonal: bool = False) -> np.ndarray:
    """
    Compute pairwise Phase Locking Value (PLV) across all channels.
    
    PLV_ij = | 1/T * sum_{t=1}^T exp(j * (phi_i(t) - phi_j(t))) |
    
    Parameters
    ----------
    eeg_window : np.ndarray
        Shape (channels, samples)
    zero_diagonal : bool
        If True, sets self-connections to 0.0, else 1.0.
        
    Returns
    -------
    plv_matrix : np.ndarray
        Shape (channels, channels) in [0, 1]
    """
    if eeg_window.ndim != 2:
        raise ValueError(f"Expected 2D array (channels, samples), got shape {eeg_window.shape}")
        
    if eeg_window.shape[1] < 2:
        raise ValueError("Window must contain at least 2 samples.")

    # Analytical signal via Hilbert Transform
    analytic = hilbert(eeg_window, axis=-1)
    phase = np.angle(analytic) # (C, T)
    
    # Vectorized computation using complex exponentials
    exp_phase = np.exp(1j * phase) # (C, T)
    
    # (C, T) @ (T, C) -> (C, C)
    plv = np.abs(np.dot(exp_phase, exp_phase.conj().T) / eeg_window.shape[1])
    plv = plv.astype(np.float32)
    
    if zero_diagonal:
        np.fill_diagonal(plv, 0.0)
    else:
        np.fill_diagonal(plv, 1.0)
        
    return np.clip(plv, 0.0, 1.0)

def compute_band_plv(eeg_window: np.ndarray,
                     sampling_rate: float = 256.0,
                     low_freq: float = 4.0,
                     high_freq: float = 8.0) -> np.ndarray:
    """
    Compute PLV in a specific neurological frequency band (e.g. Theta band 4-8 Hz).
    """
    nyq = 0.5 * sampling_rate
    b, a = butter(3, [max(low_freq / nyq, 0.001), min(high_freq / nyq, 0.99)], btype="band")
    
    band_filtered = np.zeros_like(eeg_window)
    for ch in range(eeg_window.shape[0]):
        if len(eeg_window[ch]) > 15:
            band_filtered[ch] = filtfilt(b, a, eeg_window[ch])
        else:
            band_filtered[ch] = eeg_window[ch]
            
    return compute_plv(band_filtered)
