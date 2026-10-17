# NeuroSync 4D: ST-GAT Clinical Neuro-Telemetry System (Release v2.4.2 - 2026-10-17)
"""
Continuous Wavelet Transform (CWT) and Multi-Band Power Spectral Features
"""

import numpy as np
import scipy.signal as signal

EEG_BANDS = {
    "Delta": (0.5, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta":  (13.0, 30.0),
    "Gamma": (30.0, 50.0)
}

def extract_band_power_features(eeg_window: np.ndarray, sampling_rate: float = 256.0) -> np.ndarray:
    """
    Extract relative power in Delta, Theta, Alpha, Beta, Gamma bands.
    
    Parameters
    ----------
    eeg_window : np.ndarray
        Shape (channels, samples)
    sampling_rate : float
        Sampling rate in Hz
        
    Returns
    -------
    band_powers : np.ndarray
        Shape (channels, 5) relative power per channel across 5 bands.
    """
    num_channels = eeg_window.shape[0]
    powers = np.zeros((num_channels, len(EEG_BANDS)), dtype=np.float32)
    
    nperseg = min(128, eeg_window.shape[-1])
    freqs, psd = signal.welch(eeg_window, fs=sampling_rate, nperseg=nperseg, axis=-1)
    total_power = np.sum(psd, axis=-1, keepdims=True) + 1e-9
    
    for b_idx, (band_name, (f_min, f_max)) in enumerate(EEG_BANDS.items()):
        mask = (freqs >= f_min) & (freqs <= f_max)
        if np.any(mask):
            bp = np.sum(psd[:, mask], axis=-1)
            powers[:, b_idx] = bp / total_power.squeeze(-1)
            
    return powers

def extract_cwt_spectrogram(eeg_channel: np.ndarray,
                            sampling_rate: float = 256.0,
                            num_scales: int = 32) -> np.ndarray:
    """
    Compute Continuous Wavelet Transform (CWT) scaleogram for a single channel.
    Uses PyWavelets if installed, or Morlet filterbank fallback.
    """
    try:
        import pywt
        freqs = np.geomspace(1.0, min(50.0, sampling_rate / 2.1), num=num_scales)
        scales = pywt.frequency2scale("cmor1.5-1.0", freqs / sampling_rate)
        coefs, _ = pywt.cwt(eeg_channel, scales, "cmor1.5-1.0", sampling_period=1.0 / sampling_rate)
        return np.abs(coefs).astype(np.float32)
    except Exception:
        # SciPy Morlet filterbank fallback
        widths = np.linspace(1, 32, num_scales)
        cwt_mat = signal.cwt(eeg_channel, signal.morlet2, widths)
        return np.abs(cwt_mat).astype(np.float32)
