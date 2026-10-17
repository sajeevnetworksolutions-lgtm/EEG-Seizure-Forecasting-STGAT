# NeuroSync 4D: ST-GAT Clinical Neuro-Telemetry System (Release v2.4.2 - 2026-10-17)
"""
EEG Signal Preprocessing, Digital Filtering, and Multi-Band Decomposition
"""

import numpy as np
import scipy.signal as signal
from src.config import (
    SAMPLE_RATE, BANDPASS_LOW, BANDPASS_HIGH, NOTCH_FREQ,
    EEG_BANDS, SAMPLES_PER_EPOCH, ELECTRODES_19
)

class EEGPreprocessor:
    """
    Standardized Preprocessor for multi-channel continuous Scalp EEG.
    Performs 0.5-50 Hz bandpass filtering, 50/60 Hz powerline notch filtering,
    continuous wavelet decomposition, and sliding window epoch segmentation.
    """
    def __init__(self, sample_rate=SAMPLE_RATE):
        self.sample_rate = sample_rate
        self._init_filters()

    def _init_filters(self):
        """Pre-compute Butterworth bandpass and IIR notch filter coefficients."""
        nyq = 0.5 * self.sample_rate
        low = BANDPASS_LOW / nyq
        high = min(BANDPASS_HIGH / nyq, 0.99)
        # 4th order Butterworth bandpass
        self.bp_b, self.bp_a = signal.butter(4, [low, high], btype='band')
        
        # 50 Hz / 60 Hz notch filter (Q=30)
        q = 30.0
        w0 = NOTCH_FREQ / nyq
        if w0 < 1.0:
            self.notch_b, self.notch_a = signal.iirnotch(w0, q)
        else:
            self.notch_b, self.notch_a = None, None

    def apply_filters(self, eeg_data):
        """
        Apply zero-phase bandpass and notch filters across all channels.
        Args:
            eeg_data: np.ndarray of shape (num_channels, num_samples)
        Returns:
            filtered_eeg: np.ndarray of shape (num_channels, num_samples)
        """
        filtered = np.zeros_like(eeg_data, dtype=np.float32)
        for ch in range(eeg_data.shape[0]):
            ch_sig = eeg_data[ch, :]
            # Notch filter if applicable
            if self.notch_b is not None:
                ch_sig = signal.filtfilt(self.notch_b, self.notch_a, ch_sig)
            # Bandpass filter (0.5 - 50 Hz)
            ch_sig = signal.filtfilt(self.bp_b, self.bp_a, ch_sig)
            filtered[ch, :] = ch_sig
        return filtered

    def normalize_channels(self, eeg_data):
        """
        Robust z-score normalization per channel with artifact clipping.
        Args:
            eeg_data: (num_channels, num_samples)
        Returns:
            normalized_eeg: (num_channels, num_samples)
        """
        mean = np.mean(eeg_data, axis=-1, keepdims=True)
        std = np.std(eeg_data, axis=-1, keepdims=True) + 1e-6
        norm = (eeg_data - mean) / std
        # Clip extreme motion/electrode pop artifacts to [-6, 6] std
        norm = np.clip(norm, -6.0, 6.0)
        return norm

    def extract_band_signals(self, eeg_data):
        """
        Decompose continuous EEG into 5 physiological frequency bands:
        Delta (0.5-4 Hz), Theta (4-8 Hz), Alpha (8-13 Hz), Beta (13-30 Hz), Gamma (30-50 Hz).
        Args:
            eeg_data: (num_channels, num_samples)
        Returns:
            band_dict: Dict mapping band name -> np.ndarray of shape (num_channels, num_samples)
        """
        nyq = 0.5 * self.sample_rate
        band_signals = {}
        for band_name, (low_f, high_f) in EEG_BANDS.items():
            low = max(low_f / nyq, 0.001)
            high = min(high_f / nyq, 0.99)
            b, a = signal.butter(3, [low, high], btype='band')
            band_sig = np.zeros_like(eeg_data)
            for ch in range(eeg_data.shape[0]):
                band_sig[ch, :] = signal.filtfilt(b, a, eeg_data[ch, :])
            band_signals[band_name] = band_sig
        return band_signals

    def compute_band_power(self, epoch_data):
        """
        Compute relative band power across the 5 bands for a single epoch.
        Args:
            epoch_data: (num_channels, num_samples)
        Returns:
            band_power: (num_channels, 5) relative power per channel
        """
        num_channels = epoch_data.shape[0]
        powers = np.zeros((num_channels, len(EEG_BANDS)), dtype=np.float32)
        
        freqs, psd = signal.welch(epoch_data, fs=self.sample_rate, nperseg=min(128, epoch_data.shape[-1]), axis=-1)
        total_power = np.sum(psd, axis=-1, keepdims=True) + 1e-9
        
        for b_idx, (band_name, (f_min, f_max)) in enumerate(EEG_BANDS.items()):
            mask = (freqs >= f_min) & (freqs <= f_max)
            if np.any(mask):
                bp = np.sum(psd[:, mask], axis=-1)
                powers[:, b_idx] = bp / total_power.squeeze(-1)
        return powers

    def segment_epochs(self, eeg_data, epoch_samples=SAMPLES_PER_EPOCH, overlap=0.0):
        """
        Segment continuous multi-channel EEG into discrete non-overlapping or overlapping epochs.
        Args:
            eeg_data: (num_channels, total_samples)
            epoch_samples: number of samples per epoch (e.g. 256 for 1s at 256Hz)
            overlap: fraction of overlap between epochs [0.0, 1.0)
        Returns:
            epochs: np.ndarray of shape (num_epochs, num_channels, epoch_samples)
        """
        step = int(epoch_samples * (1.0 - overlap))
        total_samples = eeg_data.shape[1]
        num_epochs = (total_samples - epoch_samples) // step + 1
        
        if num_epochs <= 0:
            return np.empty((0, eeg_data.shape[0], epoch_samples), dtype=np.float32)
            
        epochs = np.zeros((num_epochs, eeg_data.shape[0], epoch_samples), dtype=np.float32)
        for i in range(num_epochs):
            start = i * step
            end = start + epoch_samples
            epochs[i] = eeg_data[:, start:end]
            
        return epochs

    def create_temporal_sequences(self, epochs, seq_len=10):
        """
        Pack consecutive epochs into temporal sequence blocks for recurrent/GRU processing.
        Args:
            epochs: (num_epochs, num_channels, epoch_samples)
            seq_len: number of consecutive epochs per sequence
        Returns:
            sequences: (num_sequences, seq_len, num_channels, epoch_samples)
        """
        num_epochs = epochs.shape[0]
        if num_epochs < seq_len:
            # Pad if fewer epochs than sequence length
            pad_size = seq_len - num_epochs
            padding = np.repeat(epochs[-1:], pad_size, axis=0) if num_epochs > 0 else np.zeros((pad_size, epochs.shape[1], epochs.shape[2]))
            padded = np.concatenate([epochs, padding], axis=0)
            return np.expand_dims(padded, axis=0)
            
        num_seq = num_epochs - seq_len + 1
        sequences = np.zeros((num_seq, seq_len, epochs.shape[1], epochs.shape[2]), dtype=np.float32)
        for i in range(num_seq):
            sequences[i] = epochs[i:i+seq_len]
        return sequences
