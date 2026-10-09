"""
Class-Imbalanced Binary Focal Loss Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryFocalLoss(nn.Module):
    """
    Class-Imbalanced Focal Loss Formulation:
    L_FL = - alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Parameters
    ----------
    alpha : float
        Class balance weight (e.g. 0.75 for rare pre-ictal class).
    gamma : float
        Focusing parameter (e.g. 2.0 to focus on hard misclassified examples).
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "mean"):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : torch.Tensor
            Unnormalized logit predictions (B, 1) or (B,)
        targets : torch.Tensor
            Binary targets in {0.0, 1.0}
        """
        logits = logits.view(-1)
        targets = targets.view(-1).float()
        
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        
        # p_t: probability of true class
        p_t = targets * probs + (1.0 - targets) * (1.0 - probs)
        
        # alpha_t: weighting factor
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        
        # Focal modulating factor
        modulating_factor = (1.0 - p_t).pow(self.gamma)
        
        focal_loss = alpha_t * modulating_factor * bce
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
