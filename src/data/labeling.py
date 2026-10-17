# NeuroSync 4D: ST-GAT Clinical Neuro-Telemetry System (Release v2.4.2 - 2026-10-17)
"""
Scientific Seizure Forecasting Ground Truth Labeling Policy
"""

import numpy as np
from typing import List, Dict

def generate_window_labels(window_timestamps_sec: np.ndarray,
                           seizure_events: List[Dict],
                           preictal_interval_min: float = 30.0,
                           safety_gap_min: float = 0.0,
                           postictal_exclusion_min: float = 15.0) -> np.ndarray:
    """
    Assign clinical forecasting ground truth labels to EEG windows.
    
    Labels:
      0  : Inter-ictal baseline
      1  : Pre-ictal target forecasting window
     -1  : Excluded (Ictal seizure event or post-ictal recovery zone)
     
    Parameters
    ----------
    window_timestamps_sec : np.ndarray
        Array of middle/end timestamps (seconds) for each EEG window.
    seizure_events : list of dict
        List of seizures with 'start_sec' and 'end_sec'.
    preictal_interval_min : float
        Pre-ictal horizon before onset in minutes (e.g. 30.0 min = 1800s).
    safety_gap_min : float
        Safety gap before onset (e.g. 0 to 15m).
    postictal_exclusion_min : float
        Post-ictal exclusion duration after seizure offset (e.g. 15m = 900s).
    """
    labels = np.zeros(len(window_timestamps_sec), dtype=np.int32)
    
    if not seizure_events:
        return labels # All inter-ictal (0)
        
    preictal_dur = preictal_interval_min * 60.0
    gap_dur = safety_gap_min * 60.0
    postictal_dur = postictal_exclusion_min * 60.0
    
    for sz in seizure_events:
        onset = sz.get("start_sec", 0)
        offset = sz.get("end_sec", onset + 30)
        
        # 1. Pre-ictal window: [onset - preictal_dur, onset - gap_dur]
        preictal_start = max(0.0, onset - preictal_dur)
        preictal_end = max(0.0, onset - gap_dur)
        
        # 2. Ictal & Post-ictal exclusion window: [onset, offset + postictal_dur]
        exclusion_start = onset
        exclusion_end = offset + postictal_dur
        
        for idx, t in enumerate(window_timestamps_sec):
            if preictal_start <= t < preictal_end:
                labels[idx] = 1
            elif exclusion_start <= t <= exclusion_end:
                labels[idx] = -1
                
    return labels
