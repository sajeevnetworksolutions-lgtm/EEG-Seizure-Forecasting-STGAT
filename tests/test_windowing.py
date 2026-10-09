"""
Unit tests for Windowing and Temporal Formatting
"""

import pytest
import numpy as np
from src.data.windowing import segment_windows, format_subepochs

def test_segment_windows():
    # 20 seconds at 256 Hz = 5120 samples
    eeg = np.random.randn(19, 5120).astype(np.float32)
    # 10s window (2560), 5s stride (1280) -> 3 windows
    windows = segment_windows(eeg, window_samples=2560, stride_samples=1280)
    assert windows.shape == (3, 19, 2560)

def test_format_subepochs():
    window = np.random.randn(19, 2560).astype(np.float32)
    subepochs = format_subepochs(window, subepoch_samples=256)
    assert subepochs.shape == (10, 19, 256)
