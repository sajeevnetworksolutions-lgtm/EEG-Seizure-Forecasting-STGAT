# NeuroSync 4D: ST-GAT Clinical Neuro-Telemetry System (Release v2.4.2 - 2026-10-17)
"""
EEG Signal Preprocessing and Digital Filtering Pipeline
"""

import numpy as np
import scipy.signal as signal

class PreprocessingPipeline:
    """
    Standardized preprocessing pipeline for multi-channel scalp EEG signals.
    """
    def __init__(self,
                 sampling_rate: float = 256.0,
                 low_freq: float = 0.5,
                 high_freq: float = 50.0,
                 notch_freq: float = 60.0,
                 filter_order: int = 4):
        self.sampling_rate = sampling_rate
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.notch_freq = notch_freq
        self.filter_order = filter_order
        self._init_filter_coefficients()

    def _init_filter_coefficients(self):
        nyq = 0.5 * self.sampling_rate
        low = max(self.low_freq / nyq, 0.001)
        high = min(self.high_freq / nyq, 0.99)
        self.bp_b, self.bp_a = signal.butter(self.filter_order, [low, high], btype="band")
        
        # 60 Hz / 50 Hz notch filter
        w0 = self.notch_freq / nyq
        if 0 < w0 < 1.0:
            self.notch_b, self.notch_a = signal.iirnotch(w0, Q=30.0)
        else:
            self.notch_b, self.notch_a = None, None

    def filter_signal(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Apply bandpass and notch filters across all channels.
        
        Parameters
        ----------
        eeg_data : np.ndarray
            Shape (channels, samples)
            
        Returns
        -------
        filtered : np.ndarray
            Shape (channels, samples)
        """
        if eeg_data.ndim != 2:
            raise ValueError(f"Expected 2D array (channels, samples), got shape {eeg_data.shape}")
            
        filtered = np.zeros_like(eeg_data, dtype=np.float32)
        for ch in range(eeg_data.shape[0]):
            ch_data = eeg_data[ch, :]
            # Apply notch filter if applicable
            if self.notch_b is not None and len(ch_data) > 15:
                ch_data = signal.filtfilt(self.notch_b, self.notch_a, ch_data)
            # Apply bandpass filter
            if len(ch_data) > 15:
                ch_data = signal.filtfilt(self.bp_b, self.bp_a, ch_data)
            filtered[ch, :] = ch_data
            
        return filtered

    def normalize_signal(self, eeg_data: np.ndarray, method: str = "robust") -> np.ndarray:
        """
        Robust per-channel normalization:
        x' = (x - median(x)) / (IQR(x) + epsilon)
        """
        if method == "robust":
            median = np.median(eeg_data, axis=-1, keepdims=True)
            q75, q25 = np.percentile(eeg_data, [75, 25], axis=-1, keepdims=True)
            iqr = q75 - q25 + 1e-6
            norm = (eeg_data - median) / iqr
            norm = np.clip(norm, -6.0, 6.0)
        else:
            mean = np.mean(eeg_data, axis=-1, keepdims=True)
            std = np.std(eeg_data, axis=-1, keepdims=True) + 1e-6
            norm = (eeg_data - mean) / std
            norm = np.clip(norm, -6.0, 6.0)
            
        return norm.astype(np.float32)
