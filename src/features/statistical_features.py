# NeuroSync 4D: ST-GAT Clinical Neuro-Telemetry System (Release v2.4.2 - 2026-10-17)
"""
Statistical and Neurological Time-Series Feature Extraction
"""

import numpy as np
import scipy.stats as stats

def extract_hjorth_parameters(signal_1d: np.ndarray):
    """
    Compute Hjorth Activity, Mobility, and Complexity parameters.
    """
    # Activity = variance
    activity = np.var(signal_1d)
    
    # Mobility = sqrt(var(dx) / var(x))
    dx = np.diff(signal_1d)
    var_dx = np.var(dx)
    mobility = np.sqrt(var_dx / (activity + 1e-9))
    
    # Complexity = Mobility(dx) / Mobility(x)
    ddx = np.diff(dx)
    var_ddx = np.var(ddx)
    mobility_dx = np.sqrt(var_ddx / (var_dx + 1e-9))
    complexity = mobility_dx / (mobility + 1e-9)
    
    return activity, mobility, complexity

def extract_statistical_features(eeg_window: np.ndarray) -> np.ndarray:
    """
    Compute comprehensive statistical EEG features per channel.
    
    Features per channel:
    - Mean, Standard Deviation, Variance, Skewness, Kurtosis
    - Hjorth Activity, Mobility, Complexity
    - Line Length, Energy
    
    Parameters
    ----------
    eeg_window : np.ndarray
        Shape (channels, samples)
        
    Returns
    -------
    features : np.ndarray
        Shape (channels, num_features=10)
    """
    num_channels = eeg_window.shape[0]
    feats = np.zeros((num_channels, 10), dtype=np.float32)
    
    for ch in range(num_channels):
        sig = eeg_window[ch, :]
        mean_val = np.mean(sig)
        std_val = np.std(sig)
        var_val = np.var(sig)
        skew_val = float(stats.skew(sig))
        kurt_val = float(stats.kurtosis(sig))
        
        act, mob, comp = extract_hjorth_parameters(sig)
        line_length = np.sum(np.abs(np.diff(sig)))
        energy = np.sum(sig ** 2) / len(sig)
        
        feats[ch] = [mean_val, std_val, var_val, skew_val, kurt_val, act, mob, comp, line_length, energy]
        
    return feats
