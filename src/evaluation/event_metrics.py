# NeuroSync 4D: ST-GAT Clinical Neuro-Telemetry System (Release v2.4.2 - 2026-10-17)
"""
Event-Level Clinical Seizure Forecasting Evaluation
"""

import numpy as np
from typing import List, Dict

def evaluate_seizure_forecasting_events(alarm_events: List[Dict],
                                       seizure_annotations: List[Dict],
                                       total_monitoring_hours: float,
                                       min_lead_time_min: float = 5.0,
                                       max_lead_time_min: float = 30.0) -> dict:
    """
    Calculate clinical event-level forecasting sensitivity, false alarm rate, and lead time.
    
    Parameters
    ----------
    alarm_events : list of dict
        Generated alarm triggers with 'timestamp_sec'
    seizure_annotations : list of dict
        Annotated seizures with 'start_sec'
    total_monitoring_hours : float
        Total continuous duration monitored in hours
    """
    if not seizure_annotations:
        return {
            "event_sensitivity": 1.0,
            "false_alarms_per_hour": len(alarm_events) / max(total_monitoring_hours, 0.1),
            "mean_lead_time_min": 0.0,
            "forecasted_seizures": 0,
            "total_seizures": 0
        }
        
    total_seizures = len(seizure_annotations)
    forecasted_seizures = 0
    lead_times = []
    matched_alarm_indices = set()
    
    for s_idx, sz in enumerate(seizure_annotations):
        onset = sz.get("start_sec", 0)
        
        # Check if an alarm fired in [onset - max_lead, onset - min_lead]
        qualifying_alarms = []
        for a_idx, alarm in enumerate(alarm_events):
            t_alarm = alarm.get("timestamp_sec", 0)
            lead_sec = onset - t_alarm
            lead_min = lead_sec / 60.0
            
            if min_lead_time_min <= lead_min <= max_lead_time_min:
                qualifying_alarms.append((lead_min, a_idx))
                
        if qualifying_alarms:
            forecasted_seizures += 1
            # First eligible alarm determines warning lead time
            first_lead, matched_idx = max(qualifying_alarms, key=lambda x: x[0])
            lead_times.append(first_lead)
            matched_alarm_indices.add(matched_idx)
            
    false_alarms = len(alarm_events) - len(matched_alarm_indices)
    fpr_per_hour = false_alarms / max(total_monitoring_hours, 0.1)
    event_sensitivity = forecasted_seizures / total_seizures if total_seizures > 0 else 0.0
    mean_lead = float(np.mean(lead_times)) if lead_times else 0.0
    median_lead = float(np.median(lead_times)) if lead_times else 0.0
    
    return {
        "event_sensitivity": event_sensitivity,
        "false_alarms_per_hour": float(fpr_per_hour),
        "mean_lead_time_min": mean_lead,
        "median_lead_time_min": median_lead,
        "forecasted_seizures": forecasted_seizures,
        "total_seizures": total_seizures,
        "lead_time_distribution": lead_times
    }
