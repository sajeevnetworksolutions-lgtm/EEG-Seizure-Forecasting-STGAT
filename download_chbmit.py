# NeuroSync 4D: ST-GAT Clinical Neuro-Telemetry System (Release v2.4.2 - 2026-10-17)
"""
Downloader for Real CHB-MIT Scalp EEG Dataset from PhysioNet
Downloads patient summary annotations and EDF recording files.
"""

import os
import urllib.request
import sys
from pathlib import Path

DATA_RAW_DIR = Path(__file__).resolve().parent / "data" / "raw"

# PhysioNet base URL for CHB-MIT dataset
PHYSIONET_BASE_URL = "https://physionet.org/files/chbmit/1.0.0/"

# Sample files for initial validation (chb01 patient)
DEFAULT_FILES = [
    ("chb01", "chb01-summary.txt"),
    ("chb01", "chb01_01.edf"),  # Baseline inter-ictal recording (~1 hr)
    ("chb01", "chb01_03.edf"),  # Recording with verified clinical seizure at 2996s - 3036s
]

def download_file(patient_id, filename, target_dir):
    url = f"{PHYSIONET_BASE_URL}{patient_id}/{filename}"
    out_file = target_dir / filename
    
    if out_file.exists() and out_file.stat().st_size > 1000:
        print(f"[SKIP] {filename} already exists ({out_file.stat().st_size / 1e6:.2f} MB).")
        return out_file
        
    print(f"[DOWNLOADING] {url} -> {out_file} ...")
    try:
        urllib.request.urlretrieve(url, out_file)
        print(f"[DONE] Downloaded {filename} ({out_file.stat().st_size / 1e6:.2f} MB).")
        return out_file
    except Exception as e:
        print(f"[ERROR] Failed to download {filename}: {e}")
        return None

def download_chb01_sample():
    p_dir = DATA_RAW_DIR / "chb01"
    p_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Downloading Real CHB-MIT Pediatric Scalp EEG Data (Patient 01)...")
    print("=" * 60)
    
    for patient_id, filename in DEFAULT_FILES:
        download_file(patient_id, filename, p_dir)
        
    print("\nDataset download process finished.")
    print(f"Directory: {p_dir.resolve()}")

if __name__ == "__main__":
    download_chb01_sample()
