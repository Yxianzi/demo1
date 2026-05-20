"""Spatial reliability and boundary-preserving consistency losses.

All functions operate on target-domain batches only. Target labels are not
required and should not be used by these losses.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F


def build_target_coordinates(Row, Column, RandPerm, half_width):
    """Build target sample coordinates from ``utils.get_all_data`` outputs.

    Args:
        Row, Column: Non-zero padded label coordinates returned by get_all_data.
        RandPerm: Indices into Row/Column used to construct target patches.
        half_width: Padding width used by get_all_data.

    Returns:
        coords: Float tensor with shape [N, 2], ordered like the target patches.
    """

    row = np.asarray(Row)[np.asarray(RandPerm)] - int(half_width)
    col = np.asarray(Column)[np.asarray(RandPerm)] - int(half_width)
    coords = np.stack([row, col], axis=1).astype(np.float32)
    coords = np.ascontiguousarray(coords, dtype=np.float32)
    return torch.frombuffer(coords, dtype=torch.float32).reshape(coords.shape)


def spatial_sort_indices(coords):
    """Return row-major indices so adjacent target samples share batches."""

    if not torch.is_tensor(coords):
        coords = torch.as_tensor(coords)
    coords = coords.float()
    scale = coords[:, 1].max().clamp_min(0).item() + 1.0
    sort_key = coords[:, 0] * scale + coords[:, 1]
    return torch.argsort(sort_key)


def _topk_affinity(affinity, k):
    b = affinity.size(0)
    if b <= 1 or k <= 0:
        return torch.zeros_like(affinity)

    keep_k = min(int(k), b - 1)
    _, index = torch.topk(affinity, k=keep_k, dim=1, largest=True)
    mask = torch.zeros_like(affinity, dtype=torch.bool)
    mask.scatter_(1, index, True)
    affinity = affinity * mask.to(affinity.dtype)
    return torch.maximum(affinity, affinity.t())


def compute_spatial_reliability(
    probs,
    coords,
    patches,
    k=6,
    sigma_spatial=2.5,
    sigma_spectral=1.0,
):
    """Estimate target reliability from confidence and local spatial context.

    Args:
        probs: Target probabilities [B, K].
        coords: Target coordinates [B, 2].
        patches: Target HSI patches [B, C, H, W].

    Returns:
        reliability: [B]
        affinity: [B, B]
        stats: dict with reliability_mean, reliability_max, edge_num.
    """

    eps = 1e-8
    device = probs.device
    dtype = probs.dtype
    b = probs.size(0)
    confidence, pred = probs.max(dim=1)

    if b <= 1:
        reliability = confidence * 0.5
        affinity = probs.new_zeros((b, b))
        return reliability, affinity, {
            "reliability_mean": float(reliability.mean().item()) if b > 0 else 0.0,
            "reliability_max": float(reliability.max().item()) if b > 0 else 0.0,
            "edge_num": 0,
        }

    coords = coords.to(device=device, dtype=dtype)
    patches = patches.to(device=device, dtype=dtype)
    center_spectrum = patches[:, :, patches.size(2) // 2, patches.size(3) // 2]

    spatial_dist = torch.cdist(coords, coords, p=2)
    spectral_dist = torch.cdist(center_spectrum, center_spectrum, p=2)

    sigma_spatial = max(float(sigma_spatial), eps)
    sigma_spectral = max(float(sigma_spectral), eps)
    affinity_spatial = torch.exp(-(spatial_dist ** 2) / (2.0 * sigma_spatial ** 2))
    affinity_spectral = torch.exp(-(spectral_dist ** 2) / (2.0 * sigma_spectral ** 2))
    affinity = affinity_spatial * affinity_spectral
    affinity.fill_diagonal_(0.0)
    affinity = _topk_affinity(affinity, k=k)

    edge_num = int((affinity > 0).sum().item())
    if edge_num == 0:
        reliability = confidence * 0.5
        return reliability, affinity, {
            "reliability_mean": float(reliability.mean().item()),
            "reliability_max": float(reliability.max().item()),
            "edge_num": 0,
        }

    edge_sum = affinity.sum(dim=1)
    same_label = (pred.view(-1, 1) == pred.view(1, -1)).to(dtype)
    neighborhood_consistency = (affinity * same_label).sum(dim=1) / edge_sum.clamp_min(eps)

    max_possible_affinity = max(1.0, float(min(int(k), b - 1)))
    boundary = 1.0 - (edge_sum / max_possible_affinity).clamp(0.0, 1.0)
    boundary_weight = torch.exp(-boundary)
    reliability = confidence * neighborhood_consistency * boundary_weight
    reliability = torch.nan_to_num(reliability, nan=0.0, posinf=1.0, neginf=0.0)

    return reliability, affinity, {
        "reliability_mean": float(reliability.mean().item()),
        "reliability_max": float(reliability.max().item()),
        "edge_num": edge_num,
    }


def boundary_preserving_spatial_consistency(
    target_features,
    target_probs,
    coords,
    patches,
    reliability,
    k=6,
    sigma_spatial=2.5,
    sigma_spectral=1.0,
):
    """Boundary-aware prediction and feature consistency on target samples."""

    eps = 1e-8
    reliability_for_affinity, affinity, stats = compute_spatial_reliability(
        target_probs.detach(),
        coords,
        patches,
        k=k,
        sigma_spatial=sigma_spatial,
        sigma_spectral=sigma_spectral,
    )
    reliability = reliability.to(target_probs.device, target_probs.dtype)
    if reliability.numel() == 0:
        reliability = reliability_for_affinity

    weights = affinity * reliability.view(-1, 1) * reliability.view(1, -1)
    denom = weights.sum().clamp_min(eps)
    if float(denom.item()) <= eps:
        zero = target_features.new_tensor(0.0)
        return zero, zero, stats

    probs = target_probs.clamp_min(eps)
    log_probs = probs.log()
    kl_ij = (probs.unsqueeze(1) * (log_probs.unsqueeze(1) - log_probs.unsqueeze(0))).sum(dim=2)
    kl_ji = (probs.unsqueeze(0) * (log_probs.unsqueeze(0) - log_probs.unsqueeze(1))).sum(dim=2)
    symmetric_kl = 0.5 * (kl_ij + kl_ji)
    spatial_pred_loss = (weights * symmetric_kl).sum() / denom

    norm_features = F.normalize(target_features, p=2, dim=1, eps=eps)
    feature_dist = torch.cdist(norm_features, norm_features, p=2).pow(2)
    spatial_feat_loss = (weights * feature_dist).sum() / denom

    return spatial_pred_loss, spatial_feat_loss, stats
