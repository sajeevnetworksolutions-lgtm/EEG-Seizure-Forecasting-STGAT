"""
ST-GAT Model Training Pipeline with Focal Loss Optimization
"""

import os
import time
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from src.config import (
    MODEL_PARAMS, TRAIN_PARAMS, CHECKPOINT_DIR, RESULTS_DIR
)
from src.models.stgat import STGATSeizureForecaster
from src.models.losses import FocalLoss, SeizureEvaluationMetrics
from src.data.chb_mit_dataset import load_or_generate_dataset

def train_stgat(epochs=TRAIN_PARAMS["epochs"],
                batch_size=TRAIN_PARAMS["batch_size"],
                lr=TRAIN_PARAMS["learning_rate"],
                gamma=TRAIN_PARAMS["focal_gamma"],
                alpha=TRAIN_PARAMS["focal_alpha"],
                device="cpu",
                force_regenerate_data=False):
    """
    Execute full training routine for Spatio-Temporal Graph Attention Network.
    """
    device = torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")
    print(f"=== Starting ST-GAT Training on Device: {device} ===")

    # 1. Load Data
    train_loader, val_loader, test_loader, dataset_dict = load_or_generate_dataset(
        num_patients=12,
        force_regenerate=force_regenerate_data
    )
    print(f"Dataset split: Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)}")

    # 2. Initialize Model, Loss, Optimizer, Scheduler
    model = STGATSeizureForecaster().to(device)
    criterion = FocalLoss(gamma=gamma, alpha=alpha)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=TRAIN_PARAMS["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    evaluator = SeizureEvaluationMetrics(alarm_threshold=TRAIN_PARAMS["alarm_threshold"])

    best_val_auc = 0.0
    best_checkpoint_path = os.path.join(CHECKPOINT_DIR, "stgat_best_model.pt")

    history = {
        "train_loss": [], "val_loss": [],
        "val_auc_roc": [], "val_sensitivity": [], "val_specificity": []
    }

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # --- Training Phase ---
        model.train()
        train_losses = []
        for batch_x, batch_adj, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_adj = batch_adj.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits, probs = model(batch_x, batch_adj)
            loss = criterion(logits, batch_y)
            loss.backward()
            
            # Gradient clipping for stable graph attention dynamics
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()
        avg_train_loss = float(np.mean(train_losses))

        # --- Validation Phase ---
        model.eval()
        val_losses = []
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch_x, batch_adj, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_adj = batch_adj.to(device)
                batch_y = batch_y.to(device)

                logits, probs = model(batch_x, batch_adj)
                loss = criterion(logits, batch_y)
                val_losses.append(loss.item())
                
                val_preds.extend(probs.cpu().numpy().ravel())
                val_targets.extend(batch_y.cpu().numpy().ravel())

        avg_val_loss = float(np.mean(val_losses))
        val_metrics = evaluator.compute_metrics(val_targets, val_preds)
        val_auc = val_metrics["auc_roc"]
        val_sens = val_metrics["sensitivity"]
        val_spec = val_metrics["specificity"]

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_auc_roc"].append(val_auc)
        history["val_sensitivity"].append(val_sens)
        history["val_specificity"].append(val_spec)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] - "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val AUC-ROC: {val_auc:.4f} | "
              f"Sens: {val_sens*100:.1f}% | "
              f"Spec: {val_spec*100:.1f}%")

        # Checkpoint Best Model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc_roc": val_auc,
                "val_metrics": val_metrics,
                "model_params": MODEL_PARAMS
            }, best_checkpoint_path)

    elapsed = time.time() - start_time
    print(f"\nTraining Complete in {elapsed:.1f}s. Best Val AUC-ROC: {best_val_auc:.4f}")
    print(f"Saved optimal model weights to {best_checkpoint_path}")

    # Plot & Save Training History Curves
    plot_training_curves(history, os.path.join(RESULTS_DIR, "training_curves.png"))

    return best_checkpoint_path, history

def plot_training_curves(history, save_path):
    """Plot Loss and AUC-ROC curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history["train_loss"]) + 1)
    
    # Loss Curve
    ax1.plot(epochs, history["train_loss"], 'b-', label='Train Focal Loss', lw=2)
    ax1.plot(epochs, history["val_loss"], 'r--', label='Val Focal Loss', lw=2)
    ax1.set_title("Training & Validation Focal Loss", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Focal Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # AUC-ROC and Sensitivity Curve
    ax2.plot(epochs, history["val_auc_roc"], 'g-', label='Val AUC-ROC', lw=2)
    ax2.plot(epochs, history["val_sensitivity"], 'm--', label='Val Sensitivity', lw=2)
    ax2.plot(epochs, history["val_specificity"], 'c:', label='Val Specificity', lw=2)
    ax2.set_title("Validation Discrimination & Sensitivity", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Metric Score")
    ax2.set_ylim([0.0, 1.05])
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved training performance curves to {save_path}")

if __name__ == "__main__":
    train_stgat(epochs=25)
