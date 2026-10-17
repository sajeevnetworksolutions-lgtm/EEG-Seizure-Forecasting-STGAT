# NeuroSync 4D: ST-GAT Clinical Neuro-Telemetry System (Release v2.4.2 - 2026-10-17)
"""
Real-Time Inference and Clinical Alarm Controller
"""

from collections import deque
import numpy as np
import torch

class AlarmManager:
    """
    Clinical Seizure Forecasting Temporal Alarm Controller.
    
    Implements:
    - Dual-threshold hysteresis filter (activation vs clearance)
    - Consecutive persistence requirement (suppresses single-window spikes)
    - Refractory suppression interval (prevents multiple alarms for the same event)
    - Alarm history logging with timestamps, risk levels, and lead times
    """
    def __init__(self,
                 activation_threshold: float = 0.65,
                 clearance_threshold: float = 0.40,
                 persistence_windows: int = 2,
                 refractory_period_seconds: float = 1800.0): # 30 min refractory
        self.activation_threshold = activation_threshold
        self.clearance_threshold = clearance_threshold
        self.persistence_windows = persistence_windows
        self.refractory_period_seconds = refractory_period_seconds
        
        # State tracking
        self.consecutive_high_risk = 0
        self.alarm_active = False
        self.last_alarm_timestamp = -1e9
        self.recent_risk_history = deque(maxlen=60)
        self.alarm_events = []

    def update(self, risk_score: float, current_timestamp_sec: float, patient_id: str = "chb01"):
        """
        Process a new risk score observation and update alarm state.
        
        Returns
        -------
        status_dict : dict containing alarm status, label, color, and warning details.
        """
        self.recent_risk_history.append(risk_score)
        smoothed_risk = float(np.mean(list(self.recent_risk_history)[-3:])) # 3-window moving average
        
        in_refractory = (current_timestamp_sec - self.last_alarm_timestamp) < self.refractory_period_seconds
        
        triggered_now = False
        
        if risk_score >= self.activation_threshold or smoothed_risk >= self.activation_threshold:
            self.consecutive_high_risk += 1
            if self.consecutive_high_risk >= self.persistence_windows:
                if not self.alarm_active and not in_refractory:
                    self.alarm_active = True
                    self.last_alarm_timestamp = current_timestamp_sec
                    triggered_now = True
                    
                    event = {
                        "timestamp_sec": current_timestamp_sec,
                        "time_str": f"{int(current_timestamp_sec//60):02d}:{int(current_timestamp_sec%60):02d}",
                        "patient_id": patient_id,
                        "risk_score": smoothed_risk,
                        "status": "PRE-ICTAL ALARM TRIGGERED",
                        "estimated_lead_time_min": 23.4 * (1.0 - (smoothed_risk - self.activation_threshold) * 0.3)
                    }
                    self.alarm_events.append(event)
        elif risk_score < self.clearance_threshold or smoothed_risk < self.clearance_threshold:
            self.consecutive_high_risk = 0
            self.alarm_active = False

        # Status Tag & Color
        if smoothed_risk < 0.35:
            status_label = "INTER-ICTAL (NORMAL)"
            status_color = "#10b981" # Green
        elif smoothed_risk < self.activation_threshold:
            status_label = "ELEVATED VIGILANCE"
            status_color = "#f59e0b" # Amber
        elif smoothed_risk < 0.85:
            status_label = "PRE-ICTAL SEIZURE WARNING"
            status_color = "#ef4444" # Red
        else:
            status_label = "CRITICAL: IMMINENT ONSET"
            status_color = "#b91c1c" # Dark Red

        return {
            "smoothed_risk": smoothed_risk,
            "raw_risk": risk_score,
            "alarm_active": self.alarm_active,
            "triggered_now": triggered_now,
            "in_refractory": in_refractory,
            "status_label": status_label,
            "status_color": status_color,
            "total_alarms": len(self.alarm_events),
            "recent_events": self.alarm_events[-5:]
        }
