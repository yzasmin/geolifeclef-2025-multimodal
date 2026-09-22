from __future__ import annotations

import torch
from torch import nn


class MaskedAsymmetricLoss(nn.Module):
    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 2.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        if self.clip > 0:
            probs_neg = torch.clamp(1.0 - probs + self.clip, max=1.0)
        else:
            probs_neg = 1.0 - probs

        probs_pos = torch.clamp(probs, min=self.eps, max=1.0)
        probs_neg = torch.clamp(probs_neg, min=self.eps, max=1.0)

        pos_term = targets * torch.pow(1.0 - probs_pos, self.gamma_pos) * torch.log(probs_pos)
        neg_term = (1.0 - targets) * torch.pow(probs, self.gamma_neg) * torch.log(probs_neg)

        loss = -(pos_term + neg_term)
        loss = loss * mask

        denom = mask.sum().clamp(min=1.0)
        return loss.sum() / denom


def total_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    set_size_pred: torch.Tensor,
    set_size_target: torch.Tensor,
    asl_loss: MaskedAsymmetricLoss,
    set_size_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    cls_loss = asl_loss(logits, targets, mask)
    reg_loss = nn.functional.mse_loss(set_size_pred, set_size_target)
    total = cls_loss + set_size_weight * reg_loss
    return total, {
        "cls_loss": float(cls_loss.detach().cpu().item()),
        "reg_loss": float(reg_loss.detach().cpu().item()),
        "total_loss": float(total.detach().cpu().item()),
    }
