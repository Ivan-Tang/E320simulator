import torch
from torch import Tensor
import torch.nn as nn

class FocalLoss(nn.Module):
    """Binary focal loss for severe class imbalance.

    L = -α·y·(1-p)^γ·log(p) − (1-α)·(1-y)·p^γ·log(1-p)

    Defaults tuned for positive_fraction ≈ 0.0002.
    """

    def __init__(self, alpha: float = 0.995, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        eps = 1e-7
        pred = pred.clamp(eps, 1 - eps)
        ce_pos = -torch.log(pred)
        ce_neg = -torch.log(1 - pred)
        focal_pos = self.alpha * (1 - pred) ** self.gamma * ce_pos
        focal_neg = (1 - self.alpha) * pred ** self.gamma * ce_neg
        loss = target * focal_pos + (1 - target) * focal_neg
        return loss.mean()