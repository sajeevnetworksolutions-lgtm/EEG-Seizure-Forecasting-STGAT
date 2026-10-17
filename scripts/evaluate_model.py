# NeuroSync 4D: ST-GAT Clinical Neuro-Telemetry System (Release v2.4.2 - 2026-10-17)
import sys
import os
from pathlib import Path
import yaml
import numpy as np

# Windows PyTorch DLL initialization fix
if sys.platform == "win32":
    try:
        import site
        for sp in site.getsitepackages():
            torch_lib = Path(sp) / "torch" / "lib"
            if torch_lib.exists():
                os.add_dll_directory(str(torch_lib))
    except Exception:
        pass

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.st_gat import STGATSeizureForecaster
from src.evaluation.classification_metrics import compute_classification_metrics
from src.evaluation.event_metrics import evaluate_seizure_forecasting_events
from src.data.chb_mit_dataset import load_or_generate_dataset
from src.training.checkpoint import load_checkpoint

def main():
    checkpoint_path = PROJECT_ROOT / "models" / "best_model.pt"
    if not checkpoint_path.exists():
        checkpoint_path = PROJECT_ROOT / "checkpoints" / "stgat_best_model.pt"
        
    print("=" * 70)
    print(f"       CLINICAL MODEL EVALUATION: {checkpoint_path.name}")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, test_loader, _ = load_or_generate_dataset(num_patients=12)
    
    model = STGATSeizureForecaster().to(device)
    if checkpoint_path.exists():
        load_checkpoint(model, checkpoint_path, device=device)
        print(f"Loaded optimal weights from {checkpoint_path}")
        
    model.eval()
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        for batch_data in test_loader:
            x = batch_data[0].to(device)
            adj = batch_data[1].to(device)
            y = batch_data[2].to(device)
            
            _, probs = model(x, adj)
            test_preds.extend(probs.cpu().numpy().ravel())
            test_targets.extend(y.cpu().numpy().ravel())
            
    test_preds = np.array(test_preds)
    test_targets = np.array(test_targets)
    
    metrics = compute_classification_metrics(test_targets, test_preds, threshold=0.65)
    
    print("\n" + "-" * 70)
    print(f"{'Performance Metric':<35} | {'Achieved Score':<15} | {'Target'}")
    print("-" * 70)
    print(f"{'Seizure Sensitivity (%)':<35} | {metrics['sensitivity']*100:>13.2f}% | >= 94.0%")
    print(f"{'Clinical Specificity (%)':<35} | {metrics['specificity']*100:>13.2f}% | >= 90.0%")
    print(f"{'Area Under ROC Curve (AUC-ROC)':<35} | {metrics['auc_roc']:>14.4f} | >= 0.965")
    print(f"{'Area Under PR Curve (AUC-PR)':<35} | {metrics['auc_pr']:>14.4f} | --")
    print(f"{'Precision / PPV':<35} | {metrics['precision']:>14.4f} | --")
    print(f"{'F1-Score':<35} | {metrics['f1_score']:>14.4f} | --")
    print("-" * 70)
    print(f"Confusion Matrix: {metrics['confusion_matrix']}")
    print("=" * 70)

if __name__ == "__main__":
    main()
