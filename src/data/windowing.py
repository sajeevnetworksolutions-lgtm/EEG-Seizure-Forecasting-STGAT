"""
EEG Sliding Window Segmentation and Temporal Sequence Formatting
"""

import numpy as np

def segment_windows(eeg_data: np.ndarray,
                    window_samples: int = 2560,
                    stride_samples: int = 1280) -> np.ndarray:
    """
    Divide continuous multi-channel EEG into sliding windows.
    
    Parameters
    ----------
    eeg_data : np.ndarray
        Shape (channels, total_samples)
    window_samples : int
        Number of samples per window (e.g. 10s * 256Hz = 2560)
    stride_samples : int
        Stride between consecutive windows (e.g. 5s * 256Hz = 1280)
        
    Returns
    -------
    windows : np.ndarray
        Shape (num_windows, channels, window_samples)
    """
    total_samples = eeg_data.shape[1]
    if total_samples < window_samples:
        return np.empty((0, eeg_data.shape[0], window_samples), dtype=np.float32)
        
    num_windows = (total_samples - window_samples) // stride_samples + 1
    windows = np.zeros((num_windows, eeg_data.shape[0], window_samples), dtype=np.float32)
    
    for i in range(num_windows):
        start = i * stride_samples
        end = start + window_samples
        windows[i] = eeg_data[:, start:end]
        
    return windows

def format_subepochs(window: np.ndarray, subepoch_samples: int = 256) -> np.ndarray:
    """
    Divide a window (channels, window_samples) into subepochs (seq_len, channels, subepoch_samples).
    E.g. (19, 2560) -> (10, 19, 256).
    """
    num_subepochs = window.shape[1] // subepoch_samples
    subepochs = np.zeros((num_subepochs, window.shape[0], subepoch_samples), dtype=np.float32)
    
    for s in range(num_subepochs):
        start = s * subepoch_samples
        end = start + subepoch_samples
        subepochs[s] = window[:, start:end]
        
    return subepochs
