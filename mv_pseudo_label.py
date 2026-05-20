"""Lightweight multi-view pseudo-label selection for MVCPO-style training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def mvcpo_threshold(epoch, epochs, start=0.90, end=0.70):
    """Linearly decay the MVCPO confidence threshold.

    ``epoch`` is expected to start from 1. Early epochs use a higher threshold,
    while later epochs allow a slightly lower threshold.
    """

    if epochs <= 1:
        return float(end)

    progress = min(max((epoch - 1) / (epochs - 1), 0.0), 1.0)
    return float(start - (start - end) * progress)


def select_multiview_pseudo_labels(
    logits0,
    logits1,
    logits2,
    epoch,
    epochs,
    num_classes,
    threshold_start=0.90,
    threshold_end=0.70,
    pair_delta=0.05,
    pair_weight=0.7,
):
    """Select pseudo labels with lightweight three-view agreement.

    Strong samples require all three views to agree above the main threshold.
    Medium samples require a pair of views to agree above the relaxed pair
    threshold. No topology validation or prototype override is used here.
    """

    prob0_all = F.softmax(logits0, dim=1)
    prob1_all = F.softmax(logits1, dim=1)
    prob2_all = F.softmax(logits2, dim=1)

    prob0, label0 = prob0_all.max(dim=1)
    prob1, label1 = prob1_all.max(dim=1)
    prob2, label2 = prob2_all.max(dim=1)

    threshold = mvcpo_threshold(
        epoch,
        epochs,
        start=threshold_start,
        end=threshold_end,
    )
    pair_threshold = max(0.0, threshold - pair_delta)

    pseudo_labels = label0.clone()
    sample_weight = torch.zeros_like(prob0)
    candidate_mask = torch.zeros_like(label0, dtype=torch.bool)

    triple_mask = (
        (label0 == label1)
        & (label0 == label2)
        & (torch.minimum(torch.minimum(prob0, prob1), prob2) >= threshold)
    )
    pseudo_labels[triple_mask] = label0[triple_mask]
    sample_weight[triple_mask] = (
        prob0[triple_mask] + prob1[triple_mask] + prob2[triple_mask]
    ) / 3.0
    candidate_mask[triple_mask] = True

    pair01 = (
        (label0 == label1)
        & (~triple_mask)
        & (torch.minimum(prob0, prob1) >= pair_threshold)
    )
    pair02 = (
        (label0 == label2)
        & (~triple_mask)
        & (torch.minimum(prob0, prob2) >= pair_threshold)
    )
    pair12 = (
        (label1 == label2)
        & (~triple_mask)
        & (torch.minimum(prob1, prob2) >= pair_threshold)
    )

    fill = pair01 & (~candidate_mask)
    pseudo_labels[fill] = label0[fill]
    sample_weight[fill] = pair_weight * (prob0[fill] + prob1[fill]) / 2.0
    candidate_mask[fill] = True

    fill = pair02 & (~candidate_mask)
    pseudo_labels[fill] = label0[fill]
    sample_weight[fill] = pair_weight * (prob0[fill] + prob2[fill]) / 2.0
    candidate_mask[fill] = True

    fill = pair12 & (~candidate_mask)
    pseudo_labels[fill] = label1[fill]
    sample_weight[fill] = pair_weight * (prob1[fill] + prob2[fill]) / 2.0
    candidate_mask[fill] = True

    candidate_num = int(candidate_mask.sum().item())
    pair_num = int((candidate_mask & (~triple_mask)).sum().item())
    if candidate_num > 0:
        conf_mean = float(sample_weight[candidate_mask].mean().item())
        conf_max = float(sample_weight[candidate_mask].max().item())
    else:
        conf_mean = 0.0
        conf_max = 0.0

    stats = {
        "threshold": float(threshold),
        "pair_threshold": float(pair_threshold),
        "triple_num": int(triple_mask.sum().item()),
        "pair_num": pair_num,
        "candidate_num": candidate_num,
        "conf_mean": conf_mean,
        "conf_max": conf_max,
        "pseudo_hist": torch.bincount(
            pseudo_labels.detach().cpu(), minlength=num_classes
        ),
        "candidate_hist": torch.bincount(
            pseudo_labels[candidate_mask].detach().cpu(), minlength=num_classes
        ),
    }

    return pseudo_labels, candidate_mask, sample_weight, stats
