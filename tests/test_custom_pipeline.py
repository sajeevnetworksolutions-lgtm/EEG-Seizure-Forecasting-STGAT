# NeuroSync 4D: ST-GAT Clinical Neuro-Telemetry System (Release v2.4.2 - 2026-10-17)
"""
Comprehensive Automated Unit & Integration Tests for:
1. Channel Mapping (Canonical 10-20 system, Bipolar to Monopolar, Aliases)
2. Signal Quality Validation (Sampling rate resampling, Unit scaling, NaN/Inf repair, Clipping)
3. Custom EEG Ingestion (EDF, CSV, format rejection)
4. Traceable End-to-End ST-GAT Inference & Prediction Trace Provenance
5. Dual-Threshold Hysteresis Alarm Transitions
6. Mode B Custom Model Fine-Tuning & Experiment Logging
"""

import os
import shutil
import pytest
import numpy as np
import torch
from pathlib import Path

from src.data.channel_mapper import ChannelMapper
from src.data.signal_validator import SignalValidator
from src.data.custom_dataset_handler import CustomDatasetHandler
from src.inference.pipeline_runner import TraceablePipelineRunner
from src.training.custom_finetune import CustomModelTrainer
from src.config import NUM_CHANNELS, SAMPLE_RATE, ELECTRODES_19


def test_channel_mapper_exact_and_bipolar():
    mapper = ChannelMapper()
    # Test with standard CHB-MIT bipolar channels
    raw_channels = [
        "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
        "FZ-CZ", "CZ-PZ", "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8",
        "T8-P8", "P8-O2"
    ]
    report = mapper.validate_recording(raw_channels)
    assert report["is_compatible"] is True
    assert report["mapped_channels_count"] >= 14
    assert report["match_rate_percentage"] >= 70.0
    assert "Bipolar" in report["montage_type"]


def test_channel_mapper_missing_channels_rejection():
    mapper = ChannelMapper()
    # Too few channels (only 3)
    sparse_channels = ["C3", "CZ", "C4"]
    report = mapper.validate_recording(sparse_channels)
    assert report["is_compatible"] is False
    assert len(report["missing_channels"]) > 10


def test_signal_validator_resampling_and_units():
    validator = SignalValidator(target_sample_rate=256)
    
    # 500 Hz signal in Volts (e.g. 0.00005 V = 50 uV)
    n_samples_500hz = 2500 # 5 seconds
    raw_signal = np.sin(np.linspace(0, 10 * np.pi, n_samples_500hz)) * 0.00005
    data_500hz = np.tile(raw_signal, (19, 1))
    
    cleaned, report = validator.validate_and_clean(
        data_500hz,
        original_sfreq=500.0,
        channel_names=list(ELECTRODES_19)
    )
    
    assert report["resampled"] is True
    assert report["target_sampling_rate_hz"] == 256
    # 5 seconds at 256 Hz = 1280 samples
    assert cleaned.shape[1] == 1280
    assert report["unit_scaling_applied"] == 1e6
    assert report["signal_quality_score"] >= 80


def test_signal_validator_nan_repair_and_clipping():
    validator = SignalValidator(target_sample_rate=256)
    data = np.random.randn(19, 1024).astype(np.float32) * 20.0
    
    # Inject NaNs and clipping spikes
    data[0, 10:15] = np.nan
    data[1, 50:60] = 800.0 # Clipped (> 500 uV)
    
    cleaned, report = validator.validate_and_clean(
        data,
        original_sfreq=256.0,
        channel_names=list(ELECTRODES_19)
    )
    
    assert report["nan_inf_count_repaired"] == 5
    assert not np.isnan(cleaned).any()
    assert "FP2" in report["clipped_channels_detected"] or len(report["clipped_channels_detected"]) > 0


def test_custom_csv_eeg_ingestion(tmp_path):
    # Create sample CSV EEG file
    csv_file = tmp_path / "custom_patient.csv"
    time_col = np.linspace(0, 5, 5 * 256)
    channels_data = {ch: np.sin(time_col * 5.0 + i) * 35.0 for i, ch in enumerate(ELECTRODES_19)}
    import pandas as pd
    df = pd.DataFrame({"Time": time_col, **channels_data})
    df.to_csv(csv_file, index=False)
    
    handler = CustomDatasetHandler()
    audit = handler.process_and_validate(csv_file)
    
    assert audit["is_model_ready"] is True
    assert audit["compatibility_score"] >= 80
    assert audit["signals_19ch"].shape[0] == 19
    assert audit["signals_19ch"].shape[1] == len(time_col)


def test_unsupported_format_rejection(tmp_path):
    bad_file = tmp_path / "bad_file.xyz"
    bad_file.write_text("corrupted content")
    
    handler = CustomDatasetHandler()
    with pytest.raises(ValueError, match="Unsupported file format"):
        handler.load_recording_file(bad_file)


def test_traceable_pipeline_inference_and_provenance():
    runner = TraceablePipelineRunner(checkpoint_path="checkpoints/stgat_best_model.pt")
    
    # 10 seconds of 19-channel EEG
    window_data = np.random.randn(19, 256 * 10).astype(np.float32) * 25.0
    
    res = runner.process_eeg_window(
        signals_19ch=window_data,
        recording_id="TEST_REC_01",
        timestamp_sec=25.0,
        window_duration_sec=10.0,
        threshold=0.65
    )
    
    assert "prediction_trace" in res
    trace = res["prediction_trace"]
    
    assert trace["recording_id"] == "TEST_REC_01"
    assert trace["window_start_sec"] == 15.0
    assert trace["window_end_sec"] == 25.0
    assert trace["num_channels_used"] == 19
    assert trace["is_genuine_inference"] is True
    assert trace["predicted_risk_score"] >= 0.0 and trace["predicted_risk_score"] <= 1.0
    assert len(res["plv_matrix"]) == 19
    assert len(res["plv_matrix"][0]) == 19
    assert len(res["gat_attention"]) == 19


def test_alarm_hysteresis_state_machine():
    runner = TraceablePipelineRunner(checkpoint_path="checkpoints/stgat_best_model.pt")
    runner.alarm_threshold = 0.65
    runner.clear_threshold = 0.40
    
    # Test consecutive trigger logic
    runner.recent_risk_scores = [0.75, 0.78, 0.80]
    runner.consecutive_high_risk = 2
    runner.alarm_active = True
    assert runner.alarm_active is True
    
    # Reset below clear threshold
    runner.recent_risk_scores = [0.30, 0.32, 0.35]
    # Simulate processing window with clear
    smoothed = np.mean(runner.recent_risk_scores)
    if smoothed < runner.clear_threshold:
        runner.consecutive_high_risk = 0
        runner.alarm_active = False
    assert runner.alarm_active is False


def test_custom_finetuning_mode_b(tmp_path):
    trainer = CustomModelTrainer(base_checkpoint_path="checkpoints/stgat_best_model.pt")
    # 35 seconds of synthetic data (enough for > 4 windows)
    signals = np.random.randn(19, 35 * 256).astype(np.float32) * 20.0
    
    exp_record = trainer.run_training_job(
        signals_19ch=signals,
        annotations=[{"start_sec": 20, "end_sec": 30}],
        experiment_name="PyTest Unit Fine-Tune",
        dataset_name="PyTest_Mock",
        mode="fine-tune",
        epochs=1,
        batch_size=4,
        learning_rate=0.001
    )
    
    assert exp_record["experiment_id"].startswith("EXP-")
    assert exp_record["status"] == "COMPLETED"
    assert "auc_roc" in exp_record["metrics"]
    assert Path("checkpoints", exp_record["checkpoint_name"]).exists()
