# NeuroSync 4D: ST-GAT Clinical Neuro-Telemetry System (Release v2.4.2 - 2026-10-17)
"""
Custom EEG Dataset Ingestion and Validation Engine
Supports uploading and processing EDF, BDF, FIF, and CSV multi-channel EEG recordings.
"""

import os
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd
from src.data.channel_mapper import ChannelMapper
from src.data.signal_validator import SignalValidator
from src.config import SAMPLE_RATE

CUSTOM_UPLOAD_DIR = Path("data/custom_uploads")
CUSTOM_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

class CustomDatasetHandler:
    """
    Ingests arbitrary custom EEG recordings, executes channel mapping,
    signal validation, and packages data for ST-GAT model inference or training.
    """
    def __init__(self):
        self.channel_mapper = ChannelMapper()
        self.signal_validator = SignalValidator(target_sample_rate=SAMPLE_RATE)

    def load_recording_file(self, file_path: str | Path, default_sfreq: float = 256.0) -> Tuple[np.ndarray, float, List[str], Dict]:
        """
        Loads raw file based on extension (.edf, .bdf, .fif, .csv).
        Returns:
            raw_data: (num_raw_channels, num_samples)
            sfreq: sampling frequency
            ch_names: raw channel names
            file_meta: file metadata
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Recording file not found: {path}")
            
        ext = path.suffix.lower()
        
        if ext in [".edf", ".bdf"]:
            import mne
            raw = mne.io.read_raw_edf(str(path), preload=True, verbose="ERROR") if ext == ".edf" else mne.io.read_raw_bdf(str(path), preload=True, verbose="ERROR")
            raw_data = raw.get_data() # (ch, time) in Volts
            sfreq = float(raw.info["sfreq"])
            ch_names = list(raw.ch_names)
            file_meta = {
                "format": ext.upper().replace(".", ""),
                "original_channels": len(ch_names),
                "original_samples": raw_data.shape[1],
                "original_sfreq": sfreq,
                "meas_date": str(raw.info.get("meas_date", "Unknown"))
            }
        elif ext == ".fif":
            import mne
            raw = mne.io.read_raw_fif(str(path), preload=True, verbose="ERROR")
            raw_data = raw.get_data()
            sfreq = float(raw.info["sfreq"])
            ch_names = list(raw.ch_names)
            file_meta = {
                "format": "FIF",
                "original_channels": len(ch_names),
                "original_samples": raw_data.shape[1],
                "original_sfreq": sfreq
            }
        elif ext == ".csv":
            df = pd.read_csv(str(path))
            # Detect if first column is time
            first_col = df.columns[0].lower()
            if "time" in first_col or "sample" in first_col or "t" == first_col:
                time_col = df.iloc[:, 0].values
                if len(time_col) > 1 and np.ptp(time_col) > 0:
                    dt = np.median(np.diff(time_col))
                    sfreq = float(round(1.0 / dt)) if dt > 0 else default_sfreq
                else:
                    sfreq = default_sfreq
                ch_cols = df.columns[1:]
                raw_data = df[ch_cols].values.T.astype(np.float32)
                ch_names = list(ch_cols)
            else:
                sfreq = default_sfreq
                raw_data = df.values.T.astype(np.float32)
                ch_names = list(df.columns)
                
            file_meta = {
                "format": "CSV",
                "original_channels": len(ch_names),
                "original_samples": raw_data.shape[1],
                "original_sfreq": sfreq
            }
        else:
            raise ValueError(f"Unsupported file format '{ext}'. Supported: .edf, .bdf, .fif, .csv")

        return raw_data, sfreq, ch_names, file_meta

    def process_and_validate(self, file_path: str | Path, default_sfreq: float = 256.0) -> Dict:
        """
        Full ingestion, channel mapping, and signal validation pipeline.
        Returns a structured dictionary with mapped 19-channel signals and full audit reports.
        """
        raw_data, orig_sfreq, raw_ch_names, file_meta = self.load_recording_file(file_path, default_sfreq=default_sfreq)
        
        # 1. Channel Mapping to 10-20 Standard
        channel_report = self.channel_mapper.validate_recording(raw_ch_names)
        mapped_signals, target_channels = self.channel_mapper.map_signals(raw_data, raw_ch_names)
        
        # 2. Signal Validation and Cleaning (Unit check, flat channels, clipping, resampling to 256 Hz)
        cleaned_signals, signal_report = self.signal_validator.validate_and_clean(
            mapped_signals,
            original_sfreq=orig_sfreq,
            channel_names=target_channels
        )
        
        # 3. Overall Compatibility Assessment
        is_model_ready = channel_report["is_compatible"] and signal_report["signal_quality_score"] >= 40
        
        compatibility_score = int(
            channel_report["match_rate_percentage"] * 0.6 + signal_report["signal_quality_score"] * 0.4
        )
        
        return {
            "file_name": Path(file_path).name,
            "file_meta": file_meta,
            "channel_report": channel_report,
            "signal_report": signal_report,
            "is_model_ready": is_model_ready,
            "compatibility_score": compatibility_score,
            "status": "READY FOR ST-GAT FORECASTING" if is_model_ready else "INCOMPATIBLE: CHANNEL/SIGNAL DEFICIT",
            "signals_19ch": cleaned_signals, # shape (19, target_samples) at 256 Hz
            "canonical_channels": target_channels,
            "sampling_rate": SAMPLE_RATE,
            "duration_sec": signal_report["duration_seconds"]
        }
