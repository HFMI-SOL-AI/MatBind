import torch
import torch.nn as nn
import torch.nn.functional as F
from info_nce import InfoNCE
from torch import Tensor


class InfoNCELoss(nn.Module):
    def __init__(
        self,
        temperature: float,
        symmetric: bool,
        sparse: bool,
        l1_norm_weight: float,
        reduction: str = "mean",
        negative_mode: str = "unpaired",
    ):
        super().__init__()
        self.symmetric = symmetric
        self.sparse = sparse
        self.l1_norm_weight = l1_norm_weight

        self.loss_function = InfoNCE(
            temperature=temperature,
            reduction=reduction,
            negative_mode=negative_mode,
        )

    def forward(self, z1: Tensor, z2: Tensor) -> float:
        loss = self.loss_function(z1, z2)

        if self.symmetric:
            loss = (loss + self.loss_function(z2, z1)) / 2

        if self.sparse:
            loss = loss + self.l1_norm_weight * torch.norm(z1)

        return loss

class NCE(nn.Module):
    def __init__(
        self,
        symmetric: bool,
        sparse: bool,
        l1_norm_weight: float,
        reduction: str = "mean",
        temperature: float = 0.07,
    ):
        super().__init__()
        self.symmetric = symmetric
        self.sparse = sparse
        self.l1_norm_weight = l1_norm_weight
        self.reduction = reduction
        self.temperature = temperature

    def forward(self, z1: Tensor, z2: Tensor) -> Tensor:
        if z1.dim() != 2:
            raise ValueError('Embedding of modality 1 must have 2 dimensions.')
        if z2.dim() != 2:
            raise ValueError('Embedding of modality 2 must have 2 dimensions.')
        if len(z1) != len(z2):
            raise ValueError('Embedding of both modalities must must have the same number of samples.')
        if z1.shape[-1] != z2.shape[-1]:
            raise ValueError('Embedding of both modalities should have the same number of components.')

        loss = self._compute_loss(z1, z2, self.reduction)

        if self.symmetric:
            loss = (loss + self._compute_loss(z2, z1, self.reduction)) / 2

        if self.sparse:
            loss = loss + self.l1_norm_weight * torch.norm(z1)

        return loss

    def _compute_loss(self, z1: Tensor, z2: Tensor, reduction: str) -> Tensor:
        z1 = F.normalize(z1, p=2, dim=-1)
        z2 = F.normalize(z2, p=2, dim=-1)

        logits = (z1 @ transpose(z2)) / self.temperature

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(z1), device=z2.device)

        return F.cross_entropy(logits, labels, reduction=reduction)


class InfoNCEWithoutTemperature(nn.Module):
    def __init__(
        self
    ):
        super().__init__()
        self.reduction = 'mean'

    def forward(self, z1: Tensor, z2: Tensor) -> Tensor:
        if z1.dim() != 2:
            raise ValueError('Embedding of modality 1 must have 2 dimensions.')
        if z2.dim() != 2:
            raise ValueError('Embedding of modality 2 must have 2 dimensions.')
        if len(z1) != len(z2):
            raise ValueError('Embedding of both modalities must must have the same number of samples.')
        if z1.shape[-1] != z2.shape[-1]:
            raise ValueError('Embedding of both modalities should have the same number of components.')

        loss = self._compute_loss(z1, z2, self.reduction)

        loss = (loss + self._compute_loss(z2, z1, self.reduction)) / 2

        return loss

    def _compute_loss(self, z1: Tensor, z2: Tensor, reduction: str) -> Tensor:
        logits = z1 @ transpose(z2)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(z1), device=z2.device)

        return F.cross_entropy(logits, labels, reduction=reduction)

def transpose(x):
    return x.transpose(-2, -1)
