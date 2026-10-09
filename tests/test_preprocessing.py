"""
Unit tests for EEG Signal Preprocessing, Filtering, and Decomposition
"""

import pytest
import numpy as np
from src.preprocessing import EEGPreprocessor
from src.config import SAMPLE_RATE, NUM_CHANNELS, SAMPLES_PER_EPOCH, SEQUENCE_LENGTH, EEG_BANDS

@pytest.fixture
def preprocessor():
    return EEGPreprocessor(sample_rate=SAMPLE_RATE)

@pytest.fixture
def synthetic_raw_eeg():
    # 10 seconds of 19-channel EEG
    rng = np.random.RandomState(42)
    return rng.randn(NUM_CHANNELS, SAMPLE_RATE * 10).astype(np.float32)

def test_filter_output_shape_and_values(preprocessor, synthetic_raw_eeg):
    filtered = preprocessor.apply_filters(synthetic_raw_eeg)
    assert filtered.shape == synthetic_raw_eeg.shape
    assert not np.isnan(filtered).any()
    assert not np.isinf(filtered).any()

def test_channel_normalization(preprocessor, synthetic_raw_eeg):
    norm = preprocessor.normalize_channels(synthetic_raw_eeg)
    assert norm.shape == synthetic_raw_eeg.shape
    assert np.all(norm >= -6.0) and np.all(norm <= 6.0)
    # Mean should be close to 0 and std close to 1
    assert np.allclose(np.mean(norm, axis=-1), 0.0, atol=1e-1)
    assert np.allclose(np.std(norm, axis=-1), 1.0, atol=1e-1)

def test_epoch_segmentation(preprocessor, synthetic_raw_eeg):
    epochs = preprocessor.segment_epochs(synthetic_raw_eeg, epoch_samples=SAMPLES_PER_EPOCH)
    # 10 seconds / 1s epoch = 10 epochs
    assert epochs.shape == (10, NUM_CHANNELS, SAMPLES_PER_EPOCH)

def test_temporal_sequence_creation(preprocessor, synthetic_raw_eeg):
    epochs = preprocessor.segment_epochs(synthetic_raw_eeg, epoch_samples=SAMPLES_PER_EPOCH)
    seqs = preprocessor.create_temporal_sequences(epochs, seq_len=SEQUENCE_LENGTH)
    # 10 epochs with seq_len=10 gives 1 sequence block of shape (1, 10, 19, 256)
    assert seqs.shape == (1, SEQUENCE_LENGTH, NUM_CHANNELS, SAMPLES_PER_EPOCH)

def test_band_decomposition(preprocessor, synthetic_raw_eeg):
    band_dict = preprocessor.extract_band_signals(synthetic_raw_eeg)
    assert len(band_dict) == len(EEG_BANDS)
    for band_name in EEG_BANDS:
        assert band_name in band_dict
        assert band_dict[band_name].shape == synthetic_raw_eeg.shape
