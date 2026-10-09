"""
Recorded EEG Replay and Real-Time Seizure Forecasting Simulator
"""

import sys
import time
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.forecasting_engine import RealTimeForecastingEngine
from src.data.chb_mit_dataset import CHBMITDataSimulator

def replay_stream(patient_id="chb01", state="preictal", total_seconds=20, step_delay=0.1):
    print("=" * 70)
    print(f"   STARTING RECORDED EEG REPLAY STREAMING (Patient: {patient_id}, State: {state})")
    print("=" * 70)
    
    ckpt_path = PROJECT_ROOT / "models" / "best_model.pt"
    if not ckpt_path.exists():
        ckpt_path = PROJECT_ROOT / "checkpoints" / "stgat_best_model.pt"
        
    engine = RealTimeForecastingEngine(model_checkpoint_path=str(ckpt_path))
    simulator = CHBMITDataSimulator()
    
    # Generate continuous recording
    raw_eeg, _ = simulator.generate_patient_recording(patient_id, duration_sec=total_seconds, state=state)
    
    block_size = 256 # 1 second chunk (256 samples)
    num_blocks = raw_eeg.shape[1] // block_size
    
    for b in range(num_blocks):
        chunk = raw_eeg[:, b*block_size:(b+1)*block_size]
        engine.ingest_sample_block(chunk)
        
        res = engine.process_latest_window()
        timestamp = (b + 1)
        
        status_symbol = "[ALARM]" if res["alarm_active"] else "[OK]"
        lead_info = f"| Warning Lead: {res['lead_time_min']:.1f} min" if res['lead_time_min'] else ""
        print(f"T={timestamp:03d}s {status_symbol:<8} | Risk Score: {res['smoothed_risk']*100:5.1f}% | Status: {res['status_label']:<28} {lead_info}")
        time.sleep(step_delay)
        
    print("=" * 70)
    print("Replay simulation completed successfully.")

if __name__ == "__main__":
    replay_stream(patient_id="chb01", state="preictal", total_seconds=15, step_delay=0.05)
