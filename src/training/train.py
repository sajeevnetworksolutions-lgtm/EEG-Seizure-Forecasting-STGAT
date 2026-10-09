"""
Model Training Loop Pipeline with Early Stopping and Checkpointing
"""

import time
import torch
import torch.optim as optim
import numpy as np
from pathlib import Path

from src.training.validate import validate_epoch
from src.training.checkpoint import save_checkpoint

def run_training_pipeline(model: torch.nn.Module,
                          train_loader,
                          val_loader,
                          criterion: torch.nn.Module,
                          optimizer: torch.optim.Optimizer,
                          scheduler,
                          epochs: int = 30,
                          patience: int = 8,
                          checkpoint_dir: str = "models",
                          device: str = "cpu",
                          config: dict = None):
    """
    Execute full training routine.
    """
    device = torch.device(device)
    model.to(device)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_auc = 0.0
    patience_counter = 0
    history = {
        "train_loss": [], "val_loss": [],
        "val_auc_roc": [], "val_auc_pr": [],
        "val_sensitivity": [], "val_specificity": []
    }
    
    start_time = time.time()
    print(f"\n=== Training ST-GAT Model on {device} for {epochs} Epochs ===")
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        
        for batch_data in train_loader:
            x = batch_data[0].to(device)
            adj = batch_data[1].to(device)
            y = batch_data[2].to(device)
            
            optimizer.zero_grad()
            logits, probs = model(x, adj)
            loss = criterion(logits, y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            train_losses.append(loss.item())
            
        if scheduler is not None:
            scheduler.step()
            
        avg_train_loss = float(np.mean(train_losses))
        
        # Validation
        val_res = validate_epoch(model, val_loader, criterion, device)
        val_loss = val_res["val_loss"]
        val_auc = val_res["auc_roc"]
        val_sens = val_res["sensitivity"]
        val_spec = val_res["specificity"]
        
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc_roc"].append(val_auc)
        history["val_auc_pr"].append(val_res["auc_pr"])
        history["val_sensitivity"].append(val_sens)
        history["val_specificity"].append(val_spec)
        
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val AUC-ROC: {val_auc:.4f} | "
              f"Sens: {val_sens*100:.1f}% | "
              f"Spec: {val_spec*100:.1f}%")
              
        # Checkpoint Best Model
        if val_auc >= best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_res,
                config=config or {},
                save_path=checkpoint_dir / "best_model.pt"
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[EARLY STOPPING] Validation AUC-ROC did not improve for {patience} epochs.")
                break
                
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f}s. Best Val AUC-ROC: {best_val_auc:.4f}")
    return history
