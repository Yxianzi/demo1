"""Auxiliary losses for RP-MLUDA."""

import torch
import torch.nn.functional as F


def safe_supcon_loss(criterion, features, labels, mask=None):
    """Safely compute supervised contrastive loss on valid samples."""
    if mask is not None:
        sample_mask = mask.to(device=features.device, dtype=torch.bool).view(-1)
        features = features[sample_mask]
        labels = labels.to(features.device)[sample_mask]

    if features.shape[0] < 2:
        return features.new_tensor(0.0)

    return criterion(features, labels)


def symmetric_consistency_loss(logits_weak, logits_strong, mask=None):
    """Symmetric KL consistency between weak and strong predictions."""
    if mask is not None:
        sample_mask = mask.to(device=logits_weak.device, dtype=torch.bool).view(-1)
        logits_weak = logits_weak[sample_mask]
        logits_strong = logits_strong.to(logits_weak.device)[sample_mask]
    else:
        logits_strong = logits_strong.to(logits_weak.device)

    if logits_weak.shape[0] == 0:
        return logits_weak.new_tensor(0.0)

    weak_prob = F.softmax(logits_weak, dim=1).detach()
    strong_prob = F.softmax(logits_strong, dim=1)
    weak_log_prob = F.log_softmax(logits_weak, dim=1)
    strong_log_prob = F.log_softmax(logits_strong, dim=1)

    kl_weak_to_strong = F.kl_div(strong_log_prob, weak_prob, reduction="batchmean")
    kl_strong_to_weak = F.kl_div(weak_log_prob, strong_prob.detach(), reduction="batchmean")
    return 0.5 * (kl_weak_to_strong + kl_strong_to_weak)


def confidence_weighted_entropy(logits, weight=None):
    """Compute entropy minimization loss with optional confidence weights."""
    probs = F.softmax(logits, dim=1)
    log_probs = torch.log(probs.clamp_min(1e-8))
    entropy = -(probs * log_probs).sum(dim=1)

    if weight is not None:
        sample_weight = weight.to(device=logits.device, dtype=entropy.dtype).view(-1)
        entropy = entropy * sample_weight

    if entropy.numel() == 0:
        return logits.new_tensor(0.0)

    return entropy.mean()
