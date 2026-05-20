"""Spatial reliability weighted prototype alignment losses."""

import torch
import torch.nn.functional as F


def compute_class_prototypes(features, labels, num_classes):
    """Compute per-class prototypes from labeled features.

    Args:
        features: [B, D]
        labels: [B]
        num_classes: Number of classes.

    Returns:
        prototypes: [num_classes, D]
        valid_mask: [num_classes] bool
    """

    device = features.device
    dtype = features.dtype
    labels = labels.to(device=device, dtype=torch.long).view(-1)
    prototypes = features.new_zeros((num_classes, features.size(1)))
    valid_mask = torch.zeros(num_classes, device=device, dtype=torch.bool)

    for class_id in range(num_classes):
        class_mask = labels == class_id
        if class_mask.any():
            prototypes[class_id] = features[class_mask].mean(dim=0)
            valid_mask[class_id] = True

    return prototypes.to(dtype=dtype), valid_mask


def compute_reliability_weighted_prototypes(
    features,
    pseudo_labels,
    reliability,
    num_classes,
    mask=None,
):
    """Compute target prototypes weighted by spatial reliability."""

    device = features.device
    dtype = features.dtype
    pseudo_labels = pseudo_labels.to(device=device, dtype=torch.long).view(-1)
    reliability = reliability.to(device=device, dtype=dtype).view(-1)
    if mask is None:
        mask = torch.ones_like(pseudo_labels, dtype=torch.bool)
    else:
        mask = mask.to(device=device, dtype=torch.bool).view(-1)

    prototypes = features.new_zeros((num_classes, features.size(1)))
    valid_mask = torch.zeros(num_classes, device=device, dtype=torch.bool)
    eps = 1e-8

    for class_id in range(num_classes):
        class_mask = mask & (pseudo_labels == class_id)
        if class_mask.any():
            weights = reliability[class_mask].clamp_min(0.0)
            weight_sum = weights.sum()
            if weight_sum.item() > eps:
                prototypes[class_id] = (features[class_mask] * weights.view(-1, 1)).sum(dim=0) / weight_sum.clamp_min(eps)
                valid_mask[class_id] = True

    return prototypes, valid_mask


def spatial_reliable_prototype_alignment(
    source_features,
    source_labels,
    target_features,
    target_pseudo_labels,
    target_reliability,
    num_classes,
    target_mask=None,
):
    """Align source and reliable target class prototypes."""

    source_proto, source_valid = compute_class_prototypes(source_features, source_labels, num_classes)
    target_proto, target_valid = compute_reliability_weighted_prototypes(
        target_features,
        target_pseudo_labels,
        target_reliability,
        num_classes,
        mask=target_mask,
    )
    valid = source_valid & target_valid
    if not valid.any():
        return source_features.new_tensor(0.0)

    source_norm = F.normalize(source_proto[valid], p=2, dim=1, eps=1e-8)
    target_norm = F.normalize(target_proto[valid], p=2, dim=1, eps=1e-8)
    return (1.0 - (source_norm * target_norm).sum(dim=1)).mean()


def build_fused_prototypes(
    source_features,
    source_labels,
    target_features,
    target_pseudo_labels,
    target_reliability,
    num_classes,
    target_mask=None,
    eta=0.7,
):
    """Build source-target fused prototypes for the PrototypeGuidedHead."""

    source_proto, source_valid = compute_class_prototypes(source_features, source_labels, num_classes)
    target_proto, target_valid = compute_reliability_weighted_prototypes(
        target_features,
        target_pseudo_labels,
        target_reliability,
        num_classes,
        mask=target_mask,
    )
    eta = float(eta)
    fused = source_proto.clone()
    both_valid = source_valid & target_valid
    target_only = (~source_valid) & target_valid

    if both_valid.any():
        fused[both_valid] = eta * source_proto[both_valid] + (1.0 - eta) * target_proto[both_valid]
    if target_only.any():
        fused[target_only] = target_proto[target_only]

    valid = source_valid | target_valid
    fused[~valid] = 0.0
    return fused
