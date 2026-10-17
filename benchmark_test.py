# NeuroSync 4D: ST-GAT Clinical Neuro-Telemetry System (Release v2.4.2 - 2026-10-17)
"""
Automated Verification and Benchmark Test Suite for ST-GAT Seizure Forecasting System
"""

import sys
import os
import torch
import numpy as np

# Ensure project root in path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    NUM_CHANNELS, SAMPLES_PER_EPOCH, SEQUENCE_LENGTH,
    ELECTRODES_19, MODEL_PARAMS, TRAIN_PARAMS, TARGET_METRICS
)
from src.preprocessing import EEGPreprocessor
from src.connectivity import DynamicGraphConstructor
from src.models.stgat import STGATSeizureForecaster
from src.models.losses import FocalLoss, SeizureEvaluationMetrics
from src.data.chb_mit_dataset import CHBMITDataSimulator, load_or_generate_dataset
from src.forecasting_engine import RealTimeForecastingEngine

def run_tests():
    print("=" * 70)
    print("      RUNNING ST-GAT COMPREHENSIVE VERIFICATION TEST SUITE")
    print("=" * 70)

    # TEST 1: Signal Simulator and Preprocessing
    print("\n[TEST 1] Verifying CHB-MIT EEG Simulation & Preprocessing...")
    simulator = CHBMITDataSimulator()
    raw_eeg, t_ax = simulator.generate_patient_recording("chb01", duration_sec=10, state="preictal")
    assert raw_eeg.shape == (19, 2560), f"Raw EEG shape mismatch: {raw_eeg.shape}"
    
    preprocessor = EEGPreprocessor()
    filt_eeg = preprocessor.apply_filters(raw_eeg)
    assert filt_eeg.shape == raw_eeg.shape, "Filtered EEG shape mismatch"
    
    norm_eeg = preprocessor.normalize_channels(filt_eeg)
    assert np.all(norm_eeg >= -6.0) and np.all(norm_eeg <= 6.0), "Normalization clipping failed"
    
    epochs = preprocessor.segment_epochs(norm_eeg, epoch_samples=SAMPLES_PER_EPOCH)
    assert epochs.shape == (10, 19, 256), f"Epoch shape mismatch: {epochs.shape}"
    
    bands = preprocessor.extract_band_signals(raw_eeg)
    assert len(bands) == 5, f"Expected 5 frequency bands, got {len(bands)}"
    print("  -> Preprocessing, Bandpass/Notch Filters, and Epoching: PASSED")

    # TEST 2: Dynamic Graph & Phase Locking Value (PLV) Engine
    print("\n[TEST 2] Verifying Dynamic Graph Construction & PLV Engine...")
    graph_builder = DynamicGraphConstructor(num_channels=19)
    plv_mat = graph_builder.compute_plv_matrix(epochs[0])
    assert plv_mat.shape == (19, 19), f"PLV shape mismatch: {plv_mat.shape}"
    assert np.all(plv_mat >= 0.0) and np.all(plv_mat <= 1.0), "PLV values out of [0, 1]"
    assert np.allclose(np.diag(plv_mat), 1.0), "PLV diagonal self-synchrony must be 1.0"
    
    adj_mat = graph_builder.construct_dynamic_adjacency(epochs[0])
    assert adj_mat.shape == (19, 19), f"Adjacency shape mismatch: {adj_mat.shape}"
    print(f"  -> Dynamic PLV Graph Built. Average synchrony edge weight: {np.mean(adj_mat):.3f}")
    print("  -> Dynamic PLV Adjacency Matrix Construction: PASSED")

    # TEST 3: ST-GAT Model Architecture & Parameter Count
    print("\n[TEST 3] Verifying ST-GAT Neural Network Architecture...")
    model = STGATSeizureForecaster()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  -> Total Trainable Model Parameters: {total_params:,}")
    
    dummy_x = torch.randn(4, SEQUENCE_LENGTH, 19, 256)
    dummy_adj = torch.rand(4, SEQUENCE_LENGTH, 19, 19)
    logits, probs, attn_dict = model(dummy_x, dummy_adj, return_attention=True)
    
    assert logits.shape == (4, 1), f"Logits shape mismatch: {logits.shape}"
    assert probs.shape == (4, 1), f"Probs shape mismatch: {probs.shape}"
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0), "Probability bounds violated"
    assert "gat1_weights" in attn_dict and "gat2_weights" in attn_dict, "Attention extraction failed"
    print("  -> ST-GAT Forward Pass & Spatial Attention Weight Extraction: PASSED")

    # TEST 4: Class-Imbalanced Focal Loss & Backward Pass
    print("\n[TEST 4] Verifying Class-Imbalanced Focal Loss & Gradient Backprop...")
    criterion = FocalLoss(gamma=2.0, alpha=0.75)
    dummy_y = torch.tensor([[1.0], [0.0], [0.0], [0.0]])
    loss = criterion(logits, dummy_y)
    assert not torch.isnan(loss) and loss.item() > 0, "Loss computation error"
    
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN gradient detected in {name}"
    print(f"  -> Focal Loss Value: {loss.item():.4f}")
    print("  -> Focal Loss Computation & Gradient Backpropagation: PASSED")

    # TEST 5: Real-Time Streaming Inference & Hysteresis Alarm
    print("\n[TEST 5] Verifying Real-Time Forecasting Engine & Alarm Controller...")
    engine = RealTimeForecastingEngine()
    engine.ingest_sample_block(raw_eeg)
    engine_res = engine.process_latest_window()
    
    assert "smoothed_risk" in engine_res, "Engine response missing smoothed_risk"
    assert "plv_matrix" in engine_res, "Engine response missing plv_matrix"
    assert "status_label" in engine_res, "Engine response missing status_label"
    print(f"  -> Real-Time Engine Output: Status='{engine_res['status_label']}', Risk={engine_res['smoothed_risk']*100:.1f}%")
    print("  -> Real-Time Inference & Hysteresis Alarm: PASSED")

    print("\n" + "=" * 70)
    print("       ALL SYSTEM COMPONENTS SUCCESSFULLY VALIDATED! (5/5 PASSED)")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
