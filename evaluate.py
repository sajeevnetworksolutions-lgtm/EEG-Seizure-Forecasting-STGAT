"""
Comprehensive Model Evaluation and Clinical Benchmark Report Generator
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.config import CHECKPOINT_DIR, RESULTS_DIR, TARGET_METRICS, TRAIN_PARAMS
from src.models.stgat import STGATSeizureForecaster
from src.models.losses import SeizureEvaluationMetrics
from src.data.chb_mit_dataset import load_or_generate_dataset

def evaluate_model(checkpoint_path=None, device="cpu"):
    """
    Run comprehensive clinical evaluation on test cohort.
    """
    device = torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")
    if checkpoint_path is None:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "stgat_best_model.pt")

    print(f"=== Evaluating ST-GAT Model from: {checkpoint_path} ===")
    
    # Load dataset
    _, _, test_loader, dataset_dict = load_or_generate_dataset(num_patients=12)
    print(f"Test samples: {len(test_loader.dataset)}")

    # Load Model
    model = STGATSeizureForecaster().to(device)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        print(f"Loaded checkpoint trained for {ckpt.get('epoch', 'N/A')} epochs.")
    else:
        print("Warning: Checkpoint not found. Evaluating with initialized model.")
        
    model.eval()
    evaluator = SeizureEvaluationMetrics(alarm_threshold=TRAIN_PARAMS["alarm_threshold"])

    test_preds = []
    test_targets = []

    with torch.no_grad():
        for batch_x, batch_adj, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_adj = batch_adj.to(device)
            logits, probs = model(batch_x, batch_adj)
            test_preds.extend(probs.cpu().numpy().ravel())
            test_targets.extend(batch_y.numpy().ravel())

    test_preds = np.array(test_preds)
    test_targets = np.array(test_targets)

    # Compute metrics assuming ~12 hours equivalent monitoring in test set
    metrics = evaluator.compute_metrics(test_targets, test_preds, total_hours_monitored=12.0)

    # Print Clinical Benchmark Table
    print("\n" + "="*70)
    print("      CHB-MIT BENCHMARK CLINICAL EVALUATION RESULTS (ST-GAT)")
    print("="*70)
    print(f"{'Metric':<35} | {'Achieved':<15} | {'Target Synopsis'}")
    print("-"*70)
    print(f"{'Seizure Prediction Sensitivity (%)':<35} | {metrics['sensitivity']*100:>13.2f}% | >= 94.0%")
    print(f"{'False Prediction Rate (FPR/h)':<35} | {metrics['fpr_per_hour']:>14.4f} | < 0.08 / hr")
    print(f"{'Average Warning Lead Time (min)':<35} | {metrics['avg_lead_time_min']:>12.1f} min | 15 - 30 min")
    print(f"{'Area Under ROC Curve (AUC-ROC)':<35} | {metrics['auc_roc']:>14.4f} | >= 0.965")
    print(f"{'Area Under PR Curve (AUC-PR)':<35} | {metrics['auc_pr']:>14.4f} | --")
    print(f"{'Clinical Specificity (%)':<35} | {metrics['specificity']*100:>13.2f}% | >= 90.0%")
    print(f"{'F1-Score':<35} | {metrics['f1_score']:>14.4f} | --")
    print("="*70)

    # Generate Publication Figures
    generate_evaluation_plots(metrics, test_targets, test_preds)

    return metrics

def generate_evaluation_plots(metrics, y_true, y_probs):
    """Generate and save ROC, PR, and Confusion Matrix plots."""
    # 1. ROC and PR Curves Combined
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    fpr, tpr = metrics["roc_curve"]
    ax1.plot(fpr, tpr, color="#2563eb", lw=2.5, label=f"ST-GAT (AUC = {metrics['auc_roc']:.3f})")
    ax1.plot([0, 1], [0, 1], color="#9ca3af", linestyle="--", lw=1.5, label="Random Chance")
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax1.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    ax1.set_title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    recall, precision = metrics["pr_curve"]
    ax2.plot(recall, precision, color="#dc2626", lw=2.5, label=f"ST-GAT (AUC = {metrics['auc_pr']:.3f})")
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel("Recall (Sensitivity)", fontsize=11)
    ax2.set_ylabel("Precision (PPV)", fontsize=11)
    ax2.set_title("Precision-Recall Curve (Pre-Ictal Detection)", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)

    roc_path = os.path.join(RESULTS_DIR, "roc_pr_evaluation.png")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"Saved ROC & PR curves to {roc_path}")

    # 2. Confusion Matrix Heatmap
    cm = metrics["confusion_matrix"]
    matrix = np.array([[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(matrix, cmap="Blues", alpha=0.8)
    fig.colorbar(cax)
    
    for (i, j), z in np.ndenumerate(matrix):
        ax.text(j, i, f"{z:,}", ha='center', va='center', fontsize=14, fontweight="bold")
        
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Inter-Ictal (0)', 'Pre-Ictal (1)'])
    ax.set_yticklabels(['Inter-Ictal (0)', 'Pre-Ictal (1)'])
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Ground Truth", fontsize=11, fontweight="bold")
    ax.set_title("Clinical Confusion Matrix", fontsize=12, fontweight="bold", pad=20)
    
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved Confusion Matrix to {cm_path}")

if __name__ == "__main__":
    evaluate_model()
