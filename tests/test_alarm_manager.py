"""
Unit tests for Binary Focal Loss and Alarm Manager Logic
"""

import pytest
import torch
from src.models.focal_loss import BinaryFocalLoss
from src.inference.alarm_manager import AlarmManager

def test_focal_loss_value():
    criterion = BinaryFocalLoss(alpha=0.75, gamma=2.0)
    logits = torch.tensor([[-2.0], [2.0]])
    targets = torch.tensor([[0.0], [1.0]])
    loss = criterion(logits, targets)
    assert not torch.isnan(loss)
    assert loss.item() > 0.0

def test_alarm_manager_hysteresis():
    manager = AlarmManager(activation_threshold=0.65, clearance_threshold=0.40, persistence_windows=2)
    
    # 1. Low risk: no alarm
    st1 = manager.update(0.2, current_timestamp_sec=10.0)
    assert not st1["alarm_active"]
    
    # 2. First high risk spike: not yet persistent (requires 2 windows)
    st2 = manager.update(0.8, current_timestamp_sec=15.0)
    
    # 3. Second high risk: alarm activates
    st3 = manager.update(0.85, current_timestamp_sec=20.0)
    assert st3["alarm_active"]
    assert st3["triggered_now"]
    
    # 4. Moderate risk: remains active due to hysteresis
    st4 = manager.update(0.55, current_timestamp_sec=25.0)
    assert st4["alarm_active"]
    
    # 5. Low risk (< clearance_threshold): alarm clears
    st5 = manager.update(0.20, current_timestamp_sec=30.0)
    assert not st5["alarm_active"]
