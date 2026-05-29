import torch
import torch.nn.functional as F


def _zero_like_loss(reference):
    return reference.sum() * 0.0


def spatial_neighbor_preserving_loss(
    prob_t,
    feat_t=None,
    x_t=None,
    neighbor_k=4,
    sigma=1.0,
):
    """Batch spectral-neighbor prediction preserving loss.

    Args:
        prob_t: Target prediction probabilities with shape [B, C].
        feat_t: Optional fallback features when x_t is None.
        x_t: Target input patches, usually [B, bands, H, W].
        neighbor_k: Number of nearest spectral neighbors in the batch.
        sigma: Spectral similarity temperature in exp(-dist^2 / sigma).
    """

    eps = 1e-8
    batch_size = prob_t.size(0)
    if batch_size <= 1:
        return _zero_like_loss(prob_t)

    if x_t is not None:
        neighbor_base = x_t
    elif feat_t is not None:
        neighbor_base = feat_t
    else:
        neighbor_base = prob_t.detach()

    neighbor_k = min(max(int(neighbor_k), 1), batch_size - 1)
    sigma = max(float(sigma), eps)

    flat_x = neighbor_base.to(device=prob_t.device, dtype=prob_t.dtype).reshape(batch_size, -1)
    flat_x = torch.nan_to_num(flat_x, nan=0.0, posinf=0.0, neginf=0.0)
    dist = torch.cdist(flat_x, flat_x, p=2)
    dist = torch.nan_to_num(dist, nan=0.0, posinf=1e6, neginf=0.0)

    eye = torch.eye(batch_size, device=prob_t.device, dtype=torch.bool)
    dist_no_self = dist.masked_fill(eye, float("inf"))
    knn_dist, knn_idx = torch.topk(
        dist_no_self,
        k=neighbor_k,
        dim=1,
        largest=False,
    )

    prob_t = torch.nan_to_num(prob_t, nan=0.0, posinf=0.0, neginf=0.0)
    prob_i = prob_t.unsqueeze(1).expand(-1, neighbor_k, -1)
    prob_j = prob_t[knn_idx]
    prob_dist2 = (prob_i - prob_j).pow(2).sum(dim=2)

    weights = torch.exp(-knn_dist.pow(2) / sigma)
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

    valid = torch.isfinite(knn_dist)
    if not valid.any():
        return _zero_like_loss(prob_t)

    loss = (weights[valid] * prob_dist2[valid]).mean()
    return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)


def prototype_alignment_loss(
    feat_s,
    y_s,
    feat_t,
    prob_t,
    num_classes,
    conf_th=0.7,
):
    """Align source hard-label prototypes with confident target soft prototypes."""

    eps = 1e-8
    device = feat_s.device
    dtype = feat_s.dtype

    y_s = y_s.to(device=device, dtype=torch.long).view(-1)
    feat_t = feat_t.to(device=device, dtype=dtype)
    prob_t = prob_t.to(device=device, dtype=dtype)
    prob_t = torch.nan_to_num(prob_t, nan=0.0, posinf=0.0, neginf=0.0)

    if feat_s.numel() == 0 or feat_t.numel() == 0:
        return _zero_like_loss(feat_s) + _zero_like_loss(feat_t)

    max_prob, pseudo_label = prob_t.max(dim=1)
    target_mask = max_prob >= float(conf_th)
    if not target_mask.any():
        return _zero_like_loss(feat_s) + _zero_like_loss(feat_t)

    losses = []
    for class_id in range(int(num_classes)):
        source_mask = y_s == class_id
        if not source_mask.any():
            continue

        class_mask = target_mask & (pseudo_label == class_id)
        if not class_mask.any():
            continue

        class_prob = prob_t[class_mask, class_id]
        prob_sum = class_prob.sum()
        if prob_sum.item() <= eps:
            continue

        mu_s = feat_s[source_mask].mean(dim=0)
        target_feat = feat_t[class_mask]
        mu_t = (target_feat * class_prob.view(-1, 1)).sum(dim=0) / prob_sum.clamp_min(eps)
        losses.append((mu_s - mu_t).pow(2).sum())

    if len(losses) == 0:
        return _zero_like_loss(feat_s) + _zero_like_loss(feat_t)

    loss = torch.stack(losses).mean()
    return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)


def spatial_neighborhood_preserving_loss(
    target_features,
    target_logits,
    target_coords=None,
    target_patches=None,
    neighbor_k=4,
    feature_weight=0.1,
    sigma=1.0,
):
    """Compatibility wrapper for earlier SNPAN_hu.py calls."""

    _ = target_coords, feature_weight
    prob_t = F.softmax(target_logits, dim=1)
    loss = spatial_neighbor_preserving_loss(
        prob_t,
        feat_t=target_features,
        x_t=target_patches,
        neighbor_k=neighbor_k,
        sigma=sigma,
    )
    stats = {
        "spa_edges": int(prob_t.size(0) * min(max(int(neighbor_k), 0), max(prob_t.size(0) - 1, 0))),
        "spa_affinity": 0.0,
        "spa_agree": 0.0,
        "spa_conf": float(prob_t.max(dim=1)[0].mean().item()) if prob_t.numel() > 0 else 0.0,
    }
    return loss, stats


def class_prototype_alignment_loss(
    source_features,
    source_labels,
    target_features,
    target_logits,
    num_classes,
    conf_th=0.7,
):
    """Compatibility wrapper for earlier SNPAN_hu.py calls."""

    prob_t = F.softmax(target_logits, dim=1)
    loss = prototype_alignment_loss(
        source_features,
        source_labels,
        target_features,
        prob_t,
        num_classes,
        conf_th=conf_th,
    )
    with torch.no_grad():
        max_prob, pseudo_label = prob_t.max(dim=1)
        target_mask = max_prob >= float(conf_th)
        valid_classes = 0
        for class_id in range(int(num_classes)):
            has_source = (source_labels.detach().to(pseudo_label.device) == class_id).any()
            has_target = (target_mask & (pseudo_label == class_id)).any()
            if bool(has_source.item()) and bool(has_target.item()):
                valid_classes += 1
        proto_hist = torch.bincount(
            pseudo_label.detach().cpu(),
            minlength=int(num_classes),
        ).tolist()
        stats = {
            "proto_confident": int(target_mask.sum().item()),
            "proto_ratio": float(target_mask.float().mean().item()) if target_mask.numel() > 0 else 0.0,
            "proto_classes": valid_classes,
            "proto_hist": proto_hist,
        }
    return loss, stats
