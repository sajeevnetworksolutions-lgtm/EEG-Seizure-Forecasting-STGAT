# NeuroSync 4D: ST-GAT Clinical Neuro-Telemetry System (Release v2.4.2 - 2026-10-17)
"""
Unit tests for ST-GAT Model Forward Pass, Layer Shapes, and Gradient Flow
"""

import pytest
import torch
from src.models.st_gat import STGATSeizureForecaster, TemporalWaveletEncoder, SpatialGraphAttentionLayer

def test_temporal_wavelet_encoder():
    encoder = TemporalWaveletEncoder(in_samples=256, out_dim=64)
    x = torch.randn(4, 19, 256)
    out = encoder(x)
    assert out.shape == (4, 19, 64)

def test_spatial_gat_layer():
    gat = SpatialGraphAttentionLayer(in_features=64, out_features=128, num_heads=4)
    h = torch.randn(4, 19, 64)
    adj = torch.rand(4, 19, 19)
    out, attn = gat(h, adj)
    assert out.shape == (4, 19, 128)
    assert attn.shape == (4, 4, 19, 19)

def test_stgat_seizure_forecaster_forward():
    model = STGATSeizureForecaster(num_nodes=19, in_features=256)
    x = torch.randn(2, 10, 19, 256) # Batch=2, Seq=10, Channels=19, Samples=256
    adj = torch.rand(2, 10, 19, 19)
    logits, probs, attn = model(x, adj, return_attention=True)
    
    assert logits.shape == (2, 1)
    assert probs.shape == (2, 1)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)
    assert "gat1_weights" in attn
    assert "gat2_weights" in attn

def test_stgat_gradient_flow():
    model = STGATSeizureForecaster(num_nodes=19, in_features=256)
    x = torch.randn(2, 10, 19, 256)
    adj = torch.rand(2, 10, 19, 19)
    logits, _ = model(x, adj)
    target = torch.tensor([[1.0], [0.0]])
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    loss.backward()
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient missing for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
