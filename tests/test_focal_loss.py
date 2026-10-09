"""
Unit tests for Binary Focal Loss
"""

import pytest
import torch
from src.models.focal_loss import BinaryFocalLoss

def test_focal_loss():
    loss_fn = BinaryFocalLoss(alpha=0.75, gamma=2.0)
    logits = torch.randn(8, 1, requires_grad=True)
    targets = torch.randint(0, 2, (8, 1)).float()
    
    loss = loss_fn(logits, targets)
    assert loss.dim() == 0 # scalar
    assert not torch.isnan(loss)
    
    loss.backward()
    assert logits.grad is not None
