# NeuroSync 4D: ST-GAT Clinical Neuro-Telemetry System (Release v2.4.2 - 2026-10-17)
"""
Reproducible ST-GAT Training CLI Script
"""

import sys
from pathlib import Path
import yaml
import torch
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.st_gat import STGATSeizureForecaster
from src.models.focal_loss import BinaryFocalLoss
from src.training.train import run_training_pipeline
from src.data.chb_mit_dataset import load_or_generate_dataset

def main():
    # Load configuration files
    with open(PROJECT_ROOT / "configs" / "training.yaml", "r") as f:
        train_cfg = yaml.safe_load(f)
    with open(PROJECT_ROOT / "configs" / "model.yaml", "r") as f:
        model_cfg = yaml.safe_load(f)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[TRAIN SCRIPT] Starting ST-GAT model training on: {device}")
    
    # Load or generate patient-independent dataset
    train_loader, val_loader, test_loader, cache_dict = load_or_generate_dataset(num_patients=12)
    
    # Initialize Model
    model = STGATSeizureForecaster(
        num_nodes=model_cfg["model"]["num_nodes"],
        in_features=model_cfg["model"]["in_features"],
        wavelet_out_dim=model_cfg["model"]["wavelet_out_dim"],
        gat_hidden_dim=model_cfg["model"]["gat_layer_1"]["out_channels"],
        num_heads=model_cfg["model"]["gat_layer_1"]["heads"],
        gru_hidden_dim=model_cfg["model"]["temporal_gru"]["hidden_size"],
        dropout=model_cfg["model"]["gat_layer_1"]["dropout"]
    )
    
    criterion = BinaryFocalLoss(
        alpha=train_cfg["loss"]["alpha"],
        gamma=train_cfg["loss"]["gamma"]
    )
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_cfg["training"]["learning_rate"],
        weight_decay=train_cfg["training"]["weight_decay"]
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_cfg["training"]["epochs"],
        eta_min=train_cfg["optimizer"]["eta_min"]
    )
    
    history = run_training_pipeline(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=train_cfg["training"]["epochs"],
        patience=train_cfg["training"]["patience"],
        checkpoint_dir=PROJECT_ROOT / "models",
        device=device,
        config={"training": train_cfg, "model": model_cfg}
    )
    
    print("[TRAIN SCRIPT] Execution successfully completed.")

if __name__ == "__main__":
    main()
