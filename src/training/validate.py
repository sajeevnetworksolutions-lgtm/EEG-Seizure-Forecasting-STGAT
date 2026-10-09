"""
Validation Loop for ST-GAT Training
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def validate_epoch(model: torch.nn.Module,
                   val_loader,
                   criterion: torch.nn.Module,
                   device: torch.device):
    """
    Evaluate model over validation dataset.
    """
    model.eval()
    val_losses = []
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_data in val_loader:
            x = batch_data[0].to(device)
            adj = batch_data[1].to(device)
            y = batch_data[2].to(device)
            
            logits, probs = model(x, adj)
            loss = criterion(logits, y)
            
            val_losses.append(loss.item())
            all_preds.extend(probs.cpu().numpy().ravel())
            all_targets.extend(y.cpu().numpy().ravel())
            
    avg_loss = float(np.mean(val_losses))
    targets = np.array(all_targets)
    preds = np.array(all_preds)
    
    try:
        auc_roc = float(roc_auc_score(targets, preds)) if len(np.unique(targets)) > 1 else 0.5
    except Exception:
        auc_roc = 0.5
        
    try:
        auc_pr = float(average_precision_score(targets, preds)) if len(np.unique(targets)) > 1 else 0.0
    except Exception:
        auc_pr = 0.0
        
    y_pred_binary = (preds >= 0.5).astype(int)
    tp = np.sum((y_pred_binary == 1) & (targets == 1))
    fn = np.sum((y_pred_binary == 0) & (targets == 1))
    tn = np.sum((y_pred_binary == 0) & (targets == 0))
    fp = np.sum((y_pred_binary == 1) & (targets == 0))
    
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    
    return {
        "val_loss": avg_loss,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "predictions": preds,
        "targets": targets
    }
