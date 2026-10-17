# NeuroSync 4D: ST-GAT Clinical Neuro-Telemetry System (Release v2.4.2 - 2026-10-17)
"""
Window-Level Classification Performance Metrics
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, confusion_matrix
)

def compute_classification_metrics(y_true: np.ndarray,
                                   y_probs: np.ndarray,
                                   threshold: float = 0.5) -> dict:
    """
    Calculate window-level binary classification metrics.
    """
    y_true = np.asarray(y_true).ravel()
    y_probs = np.asarray(y_probs).ravel()
    
    try:
        auc_roc = float(roc_auc_score(y_true, y_probs))
    except Exception:
        auc_roc = 0.5
        
    try:
        auc_pr = float(average_precision_score(y_true, y_probs))
    except Exception:
        auc_pr = 0.0
        
    y_pred = (y_probs >= threshold).astype(int)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1 = float(2 * prec * sens / (prec + sens)) if (prec + sens) > 0 else 0.0
    
    fpr_c, tpr_c, _ = roc_curve(y_true, y_probs) if len(np.unique(y_true)) > 1 else ([0, 1], [0, 1], [])
    rec_c, prec_c, _ = precision_recall_curve(y_true, y_probs) if len(np.unique(y_true)) > 1 else ([1, 0], [0, 1], [])
    
    return {
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "f1_score": f1,
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "roc_curve": (fpr_c.tolist(), tpr_c.tolist()),
        "pr_curve": (rec_c.tolist(), prec_c.tolist())
    }
