"""
Configuration and Hyperparameters for ST-GAT Seizure Forecasting System
"""

import os
import numpy as np

# Base Paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

# Ensure output directories exist
for d in [DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# 10-20 International System 19-Channel Electrode Configuration
ELECTRODES_19 = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "O2"
]
NUM_CHANNELS = len(ELECTRODES_19)

# 2D Scalp Coordinates (Normalized to [-1, 1] for 10-20 Topographic Mapping)
# (x: Left -> Right, y: Posterior -> Anterior)
ELECTRODE_COORDS_2D = {
    "Fp1": (-0.35, 0.85),
    "Fp2": (0.35, 0.85),
    "F7":  (-0.80, 0.50),
    "F3":  (-0.40, 0.50),
    "Fz":  (0.00, 0.50),
    "F4":  (0.40, 0.50),
    "F8":  (0.80, 0.50),
    "T7":  (-0.90, 0.00),
    "C3":  (-0.45, 0.00),
    "Cz":  (0.00, 0.00),
    "C4":  (0.45, 0.00),
    "T8":  (0.90, 0.00),
    "P7":  (-0.80, -0.50),
    "P3":  (-0.40, -0.50),
    "Pz":  (0.00, -0.50),
    "P4":  (0.40, -0.50),
    "P8":  (0.80, -0.50),
    "O1":  (-0.35, -0.85),
    "O2":  (0.35, -0.85)
}

# 3D Scalp Coordinates on Unit Sphere (x, y, z)
ELECTRODE_COORDS_3D = {
    "Fp1": (-0.30, 0.80, 0.51),
    "Fp2": (0.30, 0.80, 0.51),
    "F7":  (-0.80, 0.45, 0.39),
    "F3":  (-0.40, 0.45, 0.80),
    "Fz":  (0.00, 0.45, 0.89),
    "F4":  (0.40, 0.45, 0.80),
    "F8":  (0.80, 0.45, 0.39),
    "T7":  (-0.95, 0.00, 0.31),
    "C3":  (-0.50, 0.00, 0.86),
    "Cz":  (0.00, 0.00, 1.00),
    "C4":  (0.50, 0.00, 0.86),
    "T8":  (0.95, 0.00, 0.31),
    "P7":  (-0.80, -0.45, 0.39),
    "P3":  (-0.40, -0.45, 0.80),
    "Pz":  (0.00, -0.45, 0.89),
    "P4":  (0.40, -0.45, 0.80),
    "P8":  (0.80, -0.45, 0.39),
    "O1":  (-0.30, -0.80, 0.51),
    "O2":  (0.30, -0.80, 0.51)
}

# Brain Regions Grouping
BRAIN_REGIONS = {
    "Frontal": ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8"],
    "Central": ["C3", "Cz", "C4"],
    "Temporal": ["T7", "T8"],
    "Parietal": ["P7", "P3", "Pz", "P4", "P8"],
    "Occipital": ["O1", "O2"]
}

# EEG Frequency Bands (Hz)
EEG_BANDS = {
    "Delta": (0.5, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta":  (13.0, 30.0),
    "Gamma": (30.0, 50.0)
}

# Signal Processing & Sampling Parameters
SAMPLE_RATE = 256  # Hz (CHB-MIT Standard)
EPOCH_DURATION = 1.0  # seconds per window segment
SAMPLES_PER_EPOCH = int(SAMPLE_RATE * EPOCH_DURATION)  # 256 samples
SEQUENCE_LENGTH = 10  # number of consecutive epochs fed into GRU (10 seconds temporal context)
BANDPASS_LOW = 0.5   # Hz
BANDPASS_HIGH = 50.0 # Hz
NOTCH_FREQ = 60.0    # Hz (Powerline noise)

# Clinical Forecasting Window Definitions
PRE_ICTAL_HORIZON_MIN = 15.0  # minutes before seizure
PRE_ICTAL_HORIZON_MAX = 30.0  # minutes before seizure
SEIZURE_INTERVENTION_TIME = 5.0 # minutes minimum warning required

# Model Architecture Parameters (ST-GAT)
MODEL_PARAMS = {
    "num_nodes": 19,
    "in_features": SAMPLES_PER_EPOCH, # 256 raw time-domain features per electrode per epoch
    "wavelet_out_dim": 64,            # 1D Temporal Filter output
    "gat_hidden_dim": 128,           # GAT node embedding dimension
    "num_gat_heads": 4,              # Multi-head attention heads
    "gru_hidden_dim": 128,           # Temporal sequence GRU hidden units
    "gru_num_layers": 1,
    "dropout": 0.3,
    "leak_slope": 0.2
}

# Training Hyperparameters
TRAIN_PARAMS = {
    "batch_size": 32,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 40,
    "focal_gamma": 2.0,
    "focal_alpha": 0.75,             # High alpha weighting for rare pre-ictal class (1:100 imbalance)
    "alarm_threshold": 0.65,         # Calibrated threshold for triggering pre-ictal alarm
    "patience": 8
}

# Target Clinical Benchmark Metrics
TARGET_METRICS = {
    "target_sensitivity": 0.946,      # 94.6%
    "target_fpr_per_hour": 0.08,      # < 0.08 false alarms / hr
    "target_lead_time_min": 23.4,     # 23.4 minutes warning lead time
    "target_auc_roc": 0.965           # 0.965 AUC-ROC
}

# CHB-MIT Patient Cohorts (24 Pediatric Patients)
CHB_MIT_PATIENTS = [f"chb{i:02d}" for i in range(1, 25)]
