"""Prototype memory and prototype-contrastive objective for RP-MLUDA."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class PrototypeMemory(nn.Module):
    """Maintain momentum-updated class prototypes.

    Parameters
    ----------
    num_classes:
        Number of semantic classes stored in the memory bank.
    feat_dim:
        Feature dimension of each class prototype.
    momentum:
        Exponential moving average coefficient used after a class prototype has
        been initialized.
    eps:
        Small numerical constant used for L2 normalization.
    """

    def __init__(
        self,
        num_classes: int,
        feat_dim: int,
        momentum: float = 0.9,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        if feat_dim <= 0:
            raise ValueError("feat_dim must be positive.")
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("momentum must be in [0, 1].")
        if eps <= 0:
            raise ValueError("eps must be positive.")

        self.num_classes = int(num_classes)
        self.feat_dim = int(feat_dim)
        self.momentum = float(momentum)
        self.eps = float(eps)

        self.register_buffer("prototypes", torch.zeros(num_classes, feat_dim))
        self.register_buffer("initialized", torch.zeros(num_classes, dtype=torch.bool))

    @torch.no_grad()
    def update(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> None:
        """Update class prototypes from a mini-batch.

        For each class, this method averages valid samples in the current batch.
        New classes are assigned directly, while initialized classes are updated
        with an exponential moving average. Updated prototypes are L2-normalized.
        The operation is performed without gradient tracking.
        """

        if features.dim() != 2:
            raise ValueError("features must have shape [B, D].")
        if features.size(1) != self.feat_dim:
            raise ValueError(f"features dimension must be {self.feat_dim}.")
        if labels.dim() != 1 or labels.size(0) != features.size(0):
            raise ValueError("labels must have shape [B].")
        if mask is not None and (mask.dim() != 1 or mask.size(0) != features.size(0)):
            raise ValueError("mask must have shape [B].")

        device = self.prototypes.device
        dtype = self.prototypes.dtype
        batch_features = features.detach().to(device=device, dtype=dtype)
        batch_labels = labels.detach().to(device=device, dtype=torch.long)

        if mask is None:
            valid_mask = torch.ones(batch_labels.size(0), dtype=torch.bool, device=device)
        else:
            valid_mask = mask.detach().to(device=device, dtype=torch.bool)

        valid_label_range = (batch_labels >= 0) & (batch_labels < self.num_classes)
        valid_mask = valid_mask & valid_label_range
        if not torch.any(valid_mask):
            return

        for class_idx in range(self.num_classes):
            class_mask = valid_mask & (batch_labels == class_idx)
            if not torch.any(class_mask):
                continue

            class_mean = batch_features[class_mask].mean(dim=0)
            if not self.initialized[class_idx]:
                updated = class_mean
                self.initialized[class_idx] = True
            else:
                updated = (
                    self.momentum * self.prototypes[class_idx]
                    + (1.0 - self.momentum) * class_mean
                )

            self.prototypes[class_idx] = F.normalize(
                updated.unsqueeze(0), p=2, dim=1, eps=self.eps
            ).squeeze(0)

    def get(self) -> torch.Tensor:
        """Return the current prototype tensor with shape ``[num_classes, feat_dim]``."""

        return self.prototypes


def compute_prototype_prediction(features, prototypes):
    norm_features = F.normalize(features, p=2, dim=1, eps=1e-8)
    norm_prototypes = F.normalize(
        prototypes.to(device=features.device, dtype=features.dtype), p=2, dim=1, eps=1e-8
    )
    proto_logits = torch.matmul(norm_features, norm_prototypes.t())
    proto_logits = torch.where(torch.isfinite(proto_logits), proto_logits, torch.zeros_like(proto_logits))
    proto_pred = proto_logits.argmax(dim=1)
    return proto_logits, proto_pred


def class_balanced_topk_mask(labels, scores, candidate_mask, num_classes, top_k):
    device = labels.device
    balanced_mask = torch.zeros(labels.size(0), dtype=torch.bool, device=device)
    if top_k <= 0:
        return balanced_mask

    labels = labels.to(device=device, dtype=torch.long).view(-1)
    scores = scores.to(device=device).view(-1)
    candidate_mask = candidate_mask.to(device=device, dtype=torch.bool).view(-1)

    for class_idx in range(num_classes):
        class_mask = candidate_mask & (labels == class_idx)
        if not torch.any(class_mask):
            continue

        class_indices = torch.nonzero(class_mask, as_tuple=False).view(-1)
        k = min(int(top_k), class_indices.numel())
        top_indices = torch.topk(scores[class_indices], k=k, largest=True).indices
        balanced_mask[class_indices[top_indices]] = True

    return balanced_mask


def confidence_weighted_prototype_loss(features, labels, prototypes, confidence, temperature=0.1, mask=None):
    if mask is not None:
        valid_mask = mask.to(device=features.device, dtype=torch.bool).view(-1)
        if not torch.any(valid_mask):
            return features.new_zeros(())
        features = features[valid_mask]
        labels = labels.to(device=features.device, dtype=torch.long)[valid_mask]
        confidence = confidence.to(device=features.device, dtype=features.dtype)[valid_mask]
    else:
        labels = labels.to(device=features.device, dtype=torch.long)
        confidence = confidence.to(device=features.device, dtype=features.dtype)

    if features.size(0) == 0:
        return features.new_zeros(())

    safe_temperature = max(float(temperature), 1e-8)
    norm_features = F.normalize(features, p=2, dim=1, eps=1e-8)
    norm_prototypes = F.normalize(
        prototypes.to(device=features.device, dtype=features.dtype), p=2, dim=1, eps=1e-8
    )
    logits = torch.matmul(norm_features, norm_prototypes.t()) / safe_temperature
    logits = torch.where(torch.isfinite(logits), logits, torch.zeros_like(logits))
    loss_each = F.cross_entropy(logits, labels, reduction="none")
    confidence = torch.where(torch.isfinite(confidence), confidence, torch.zeros_like(confidence))
    loss = torch.sum(loss_each * confidence) / (torch.sum(confidence) + 1e-8)
    if torch.isfinite(loss).item():
        return loss
    return features.new_zeros(())


def prototype_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    temperature: float = 0.1,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute classification-style contrastive loss to class prototypes.

    Features and prototypes are L2-normalized before computing logits. If a mask
    is provided, only samples with ``mask=True`` contribute to the loss. Empty
    valid samples produce a scalar zero tensor on the same device and dtype as
    ``features``. Zero-valued prototypes are allowed and remain numerically safe.
    """

    if features.dim() != 2:
        raise ValueError("features must have shape [B, D].")
    if prototypes.dim() != 2 or prototypes.size(1) != features.size(1):
        raise ValueError("prototypes must have shape [C, D].")
    if labels.dim() != 1 or labels.size(0) != features.size(0):
        raise ValueError("labels must have shape [B].")
    if mask is not None and (mask.dim() != 1 or mask.size(0) != features.size(0)):
        raise ValueError("mask must have shape [B].")

    if mask is not None:
        valid_mask = mask.to(device=features.device, dtype=torch.bool)
        if not torch.any(valid_mask):
            return features.new_zeros(())
        features = features[valid_mask]
        labels = labels.to(device=features.device, dtype=torch.long)[valid_mask]
    else:
        labels = labels.to(device=features.device, dtype=torch.long)

    if features.size(0) == 0:
        return features.new_zeros(())

    safe_temperature = max(float(temperature), 1e-8)
    norm_features = F.normalize(features, p=2, dim=1, eps=1e-8)
    norm_prototypes = F.normalize(
        prototypes.to(device=features.device, dtype=features.dtype), p=2, dim=1, eps=1e-8
    )
    logits = torch.matmul(norm_features, norm_prototypes.t()) / safe_temperature
    return F.cross_entropy(logits, labels)
