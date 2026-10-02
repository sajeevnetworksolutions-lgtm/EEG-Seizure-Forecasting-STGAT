"""
Real-Time Seizure Forecasting Engine and Clinical Alarm Controller
"""

import numpy as np
import torch
from collections import deque
from src.config import (
    NUM_CHANNELS, SAMPLE_RATE, SAMPLES_PER_EPOCH, SEQUENCE_LENGTH,
    ELECTRODES_19, BRAIN_REGIONS
)
from src.preprocessing import EEGPreprocessor
from src.connectivity import DynamicGraphConstructor
from src.models.stgat import STGATSeizureForecaster

class RealTimeForecastingEngine:
    """
    Hospital-grade continuous real-time EEG streaming inference engine.
    Maintains a rolling ring buffer of multi-channel EEG, applies dynamic PLV graph construction,
    runs ST-GAT model inference, and manages calibrated dual-threshold alarm triggering.
    """
    def __init__(self, model_checkpoint_path=None, alarm_threshold=0.65, clear_threshold=0.40, device="cpu"):
        self.device = torch.device(device)
        self.alarm_threshold = alarm_threshold
        self.clear_threshold = clear_threshold
        
        # Signal processing components
        self.preprocessor = EEGPreprocessor(sample_rate=SAMPLE_RATE)
        self.graph_builder = DynamicGraphConstructor(num_channels=NUM_CHANNELS)
        
        # Load or initialize ST-GAT model
        self.model = STGATSeizureForecaster().to(self.device)
        if model_checkpoint_path and os.path.exists(model_checkpoint_path):
            try:
                ckpt = torch.load(model_checkpoint_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(ckpt.get("model_state_dict", ckpt))
                print(f"Loaded ST-GAT model weights from {model_checkpoint_path}")
            except Exception as e:
                print(f"Warning loading checkpoint: {e}. Running with initialized weights.")
        self.model.eval()

        # Ring buffers for continuous stream
        self.buffer_size_samples = SAMPLE_RATE * (SEQUENCE_LENGTH + 2) # e.g. 12 seconds
        self.raw_buffer = np.zeros((NUM_CHANNELS, self.buffer_size_samples), dtype=np.float32)
        
        # State tracking
        self.recent_risk_scores = deque(maxlen=30)
        self.alarm_active = False
        self.alarm_history = []
        self.consecutive_high_risk = 0
        self.estimated_time_to_onset_sec = None

    def ingest_sample_block(self, new_samples):
        """
        Push newly arrived EEG sample block into ring buffer.
        Args:
            new_samples: np.ndarray of shape (19, block_size)
        """
        block_len = new_samples.shape[1]
        if block_len >= self.buffer_size_samples:
            self.raw_buffer = new_samples[:, -self.buffer_size_samples:]
        else:
            self.raw_buffer = np.roll(self.raw_buffer, -block_len, axis=1)
            self.raw_buffer[:, -block_len:] = new_samples

    def process_latest_window(self):
        """
        Extract the most recent SEQUENCE_LENGTH window from buffer, preprocess,
        compute dynamic PLV graphs, and execute ST-GAT inference.
        Returns:
            result: Dict containing risk_score, alarm_status, lead_time_min,
                    attention_weights, plv_matrix, band_power, and regional_activity
        """
        # Extract last SEQUENCE_LENGTH seconds
        window_samples = SEQUENCE_LENGTH * SAMPLES_PER_EPOCH
        raw_window = self.raw_buffer[:, -window_samples:]
        
        # Step 1: Preprocess (Bandpass filter + Normalization)
        filtered_window = self.preprocessor.apply_filters(raw_window)
        norm_window = self.preprocessor.normalize_channels(filtered_window)
        
        # Step 2: Epoch segmentation
        epochs = self.preprocessor.segment_epochs(norm_window, epoch_samples=SAMPLES_PER_EPOCH)
        if epochs.shape[0] < SEQUENCE_LENGTH:
            # Pad if not yet filled
            pad_n = SEQUENCE_LENGTH - epochs.shape[0]
            epochs = np.pad(epochs, ((pad_n, 0), (0, 0), (0, 0)), mode='edge')
            
        seq_input = epochs[:SEQUENCE_LENGTH] # (10, 19, 256)
        
        # Step 3: Dynamic PLV Graph Construction
        adjs = self.graph_builder.construct_batch_adjacencies(seq_input[np.newaxis, ...]) # (1, 10, 19, 19)
        latest_plv = self.graph_builder.compute_plv_matrix(epochs[-1])
        
        # Step 4: Multi-band Power Spectral Density
        band_power = self.preprocessor.compute_band_power(epochs[-1]) # (19, 5)
        
        # Step 5: ST-GAT Model Inference
        x_tensor = torch.tensor(seq_input[np.newaxis, ...], dtype=torch.float32).to(self.device)
        adj_tensor = torch.tensor(adjs, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            logits, probs, attn_dict = self.model(x_tensor, adj_tensor, return_attention=True)
            
        raw_risk = float(probs.cpu().numpy()[0, 0])
        self.recent_risk_scores.append(raw_risk)
        
        # Smoothed risk score (moving average of last 3 windows)
        smoothed_risk = float(np.mean(list(self.recent_risk_scores)[-3:]))
        
        # Step 6: Dual-Threshold Hysteresis Alarm Controller
        if smoothed_risk >= self.alarm_threshold:
            self.consecutive_high_risk += 1
            if self.consecutive_high_risk >= 2: # Requires 2 consecutive high risk epochs
                self.alarm_active = True
                if self.estimated_time_to_onset_sec is None:
                    # Initialize warning countdown: ~23.4 minutes nominal lead time calibrated with risk
                    lead_min = 23.4 * (1.0 - (smoothed_risk - self.alarm_threshold) * 0.4)
                    self.estimated_time_to_onset_sec = max(10.0, lead_min) * 60.0
        elif smoothed_risk < self.clear_threshold:
            self.consecutive_high_risk = 0
            self.alarm_active = False
            self.estimated_time_to_onset_sec = None

        # Decrement countdown if alarm active
        if self.alarm_active and self.estimated_time_to_onset_sec is not None:
            self.estimated_time_to_onset_sec = max(0.0, self.estimated_time_to_onset_sec - 1.0)
            
        # Step 7: Regional Activity & Node Importance
        # Extract mean GAT attention weights per node
        gat_attn = attn_dict["gat2_weights"].squeeze(0).mean(dim=0).cpu().numpy() # (19, 19)
        node_importance = np.mean(gat_attn, axis=0) # (19,)
        node_importance = (node_importance - np.min(node_importance)) / (np.ptp(node_importance) + 1e-6)
        
        # Compute regional risk contribution
        region_scores = {}
        for region_name, ch_list in BRAIN_REGIONS.items():
            indices = [ELECTRODES_19.index(ch) for ch in ch_list if ch in ELECTRODES_19]
            if indices:
                region_scores[region_name] = float(np.mean(node_importance[indices]))
            else:
                region_scores[region_name] = 0.0

        # Clinical Status Tag
        if smoothed_risk < 0.35:
            status_label = "INTER-ICTAL (NORMAL)"
            status_color = "#10b981" # Green
        elif smoothed_risk < 0.65:
            status_label = "ELEVATED VIGILANCE"
            status_color = "#f59e0b" # Yellow / Amber
        elif smoothed_risk < 0.85:
            status_label = "PRE-ICTAL SEIZURE WARNING"
            status_color = "#ef4444" # Red
        else:
            status_label = "CRITICAL: IMMINENT ONSET"
            status_color = "#b91c1c" # Dark Red / Flashing

        return {
            "raw_risk": raw_risk,
            "smoothed_risk": smoothed_risk,
            "alarm_active": self.alarm_active,
            "status_label": status_label,
            "status_color": status_color,
            "lead_time_min": (self.estimated_time_to_onset_sec / 60.0) if self.estimated_time_to_onset_sec else None,
            "plv_matrix": latest_plv,
            "gat_attention": gat_attn,
            "node_importance": node_importance,
            "band_power": band_power,
            "region_scores": region_scores,
            "recent_eeg": norm_window[:, -SAMPLES_PER_EPOCH*4:] # last 4 seconds for display
        }
