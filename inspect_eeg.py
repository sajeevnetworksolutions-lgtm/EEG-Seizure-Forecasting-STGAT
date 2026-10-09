"""
Diagnostic and Inspection Tool for Real CHB-MIT Scalp EEG EDF Recordings
"""

import os
import sys
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ELECTRODES_19

def parse_chb_summary(summary_path):
    """
    Parse CHB-MIT summary text file for seizure annotations.
    """
    if not summary_path.exists():
        return {}
        
    seizure_info = {}
    current_file = None
    
    with open(summary_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if line.startswith("File Name:"):
            current_file = line.split(":")[-1].strip()
            seizure_info[current_file] = {"num_seizures": 0, "seizures": []}
        elif line.startswith("Number of Seizures in File:") and current_file:
            num = int(line.split(":")[-1].strip())
            seizure_info[current_file]["num_seizures"] = num
        elif line.startswith("Seizure") and "Start Time:" in line and current_file:
            start_sec = int(line.split(":")[-1].replace("seconds", "").strip())
            seizure_info[current_file]["seizures"].append({"start_sec": start_sec})
        elif line.startswith("Seizure") and "End Time:" in line and current_file:
            end_sec = int(line.split(":")[-1].replace("seconds", "").strip())
            if seizure_info[current_file]["seizures"]:
                seizure_info[current_file]["seizures"][-1]["end_sec"] = end_sec
                
    return seizure_info

def inspect_edf(edf_path, summary_path=None):
    """
    Inspect real EDF recording metadata using MNE-Python.
    """
    edf_path = Path(edf_path)
    if not edf_path.exists():
        print(f"[ERROR] EEG recording not found: {edf_path.resolve()}")
        return None

    try:
        import mne
    except ImportError:
        print("[ERROR] MNE-Python is required. Install via: pip install mne")
        return None

    print("=" * 70)
    print(f"      INSPECTING REAL CLINICAL EEG RECORDING: {edf_path.name}")
    print("=" * 70)

    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
    
    sfreq = raw.info["sfreq"]
    n_channels = len(raw.ch_names)
    ch_names = raw.ch_names
    duration_sec = raw.times[-1]
    duration_hrs = duration_sec / 3600.0

    print(f"• File Path:             {edf_path.resolve()}")
    print(f"• Sampling Frequency:    {sfreq} Hz")
    print(f"• Number of Channels:    {n_channels}")
    print(f"• Total Duration:        {duration_sec:.1f} seconds ({duration_hrs:.2f} hours)")
    print(f"• Channel Names ({len(ch_names)}):")
    print("  " + ", ".join(ch_names[:10]) + ("..." if len(ch_names) > 10 else ""))

    # Check mapping to 10-20 19-Electrode System
    print("\n--- 10-20 Standard Electrode Compatibility Check ---")
    matched_channels = []
    for std_ch in ELECTRODES_19:
        # Match standard or bipolar (e.g. FP1-F7, FP1-F3, etc.)
        matches = [c for c in ch_names if std_ch.upper() in c.upper()]
        if matches:
            matched_channels.append((std_ch, matches[0]))
            
    print(f"• Matched {len(matched_channels)} / {len(ELECTRODES_19)} 10-20 standard electrode channels.")

    # Summary Seizure Annotations
    if summary_path and Path(summary_path).exists():
        print("\n--- Clinical Seizure Annotations ---")
        summary_info = parse_chb_summary(Path(summary_path))
        file_info = summary_info.get(edf_path.name, None)
        if file_info:
            print(f"• Number of Verified Seizures: {file_info['num_seizures']}")
            for idx, sz in enumerate(file_info["seizures"], 1):
                s_start = sz.get("start_sec", 0)
                s_end = sz.get("end_sec", 0)
                dur = s_end - s_start
                print(f"  - Seizure #{idx}: Onset at {s_start}s ({s_start/60:.1f} min) -> Offset at {s_end}s | Duration: {dur}s")
        else:
            print("• Recording classified as Inter-Ictal Baseline (No clinical seizure onsets).")

    print("=" * 70)
    return raw

if __name__ == "__main__":
    raw_dir = Path("data/raw/chb01")
    edf_target = raw_dir / "chb01_03.edf" if (raw_dir / "chb01_03.edf").exists() else raw_dir / "chb01_01.edf"
    sum_target = raw_dir / "chb01-summary.txt"
    inspect_edf(edf_target, sum_target)
