# NeuroSync 4D: ST-GAT Clinical Neuro-Telemetry System (Release v2.4.2 - 2026-10-17)
"""
Unit tests for EDF Loading and Channel Mapping
"""

import pytest
import numpy as np
from pathlib import Path
from src.data.load_edf import get_channel_mapping
from src.config import ELECTRODES_19

def test_channel_mapping_bipolar():
    raw_names = [
        "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
        "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
        "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
        "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
        "FZ-CZ", "CZ-PZ", "T7-FT9", "FT9-FT10"
    ]
    mapping = get_channel_mapping(raw_names, ELECTRODES_19)
    assert "Fp1" in mapping
    assert "F7" in mapping
    assert "C3" in mapping
    assert len(mapping) >= 15

def test_channel_mapping_monopolar():
    raw_names = [f"EEG {ch}" for ch in ELECTRODES_19]
    mapping = get_channel_mapping(raw_names, ELECTRODES_19)
    assert len(mapping) == 19
