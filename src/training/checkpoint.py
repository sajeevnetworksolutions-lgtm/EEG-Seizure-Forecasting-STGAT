"""
Model Checkpointing and State Persistence Module
"""

import os
from pathlib import Path
import torch

def save_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    metrics: dict,
                    config: dict,
                    save_path: str | Path):
    """
    Save complete training state checkpoint.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": config
    }
    torch.save(state, save_path)
    print(f"[CHECKPOINT] Saved state to {save_path.resolve()}")

def load_checkpoint(model: torch.nn.Module,
                    checkpoint_path: str | Path,
                    optimizer: torch.optim.Optimizer = None,
                    device: str = "cpu"):
    """
    Load model checkpoint.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path.resolve()}")
        
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        
    return ckpt
