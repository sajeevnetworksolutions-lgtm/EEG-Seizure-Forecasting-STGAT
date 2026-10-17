# NeuroSync 4D: ST-GAT Clinical Neuro-Telemetry System (Release v2.4.2 - 2026-10-17)
"""
Unit tests for Clinical DatabaseManager and Multi-Patient Loader
"""

import pytest
import numpy as np
from pathlib import Path
from src.data.database_manager import DatabaseManager, DATABASES, PATIENT_PROFILES
from src.data.chb_mit_dataset import CHBMITDataSimulator

def test_database_manager_metadata():
    dbs = DatabaseManager.get_all_databases()
    assert "CHB_MIT" in dbs
    assert "SIENA_SCALP" in dbs
    assert "BONN_BENCHMARK" in dbs
    
    assert len(dbs["CHB_MIT"]["patients"]) >= 3

def test_patient_profiles_completeness():
    patients = DatabaseManager.get_all_patients()
    assert "chb01" in patients
    assert "chb02" in patients
    assert "chb03" in patients
    
    for pid in ["chb01", "chb02", "chb03"]:
        prof = DatabaseManager.get_patient_profile(pid)
        assert "focal_channels" in prof
        assert len(prof["focal_channels"]) > 0
        assert "metrics" in prof
        assert prof["metrics"]["sensitivity"] >= 80.0
        assert prof["metrics"]["auc_roc"] >= 0.85

def test_multi_patient_data_loader():
    sim = CHBMITDataSimulator()
    for pid in ["chb01", "chb02", "chb03"]:
        eeg, t = sim.load_real_patient_recording(pid, duration_sec=4)
        assert eeg.shape == (19, 4 * 256)
        assert len(t) == 4 * 256
        assert not np.isnan(eeg).any()
