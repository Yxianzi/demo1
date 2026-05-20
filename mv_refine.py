"""Multi-view pseudo-label refinement for MLUDA."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def multiview_refine_pseudo_labels(
    logits0,
    logits1,
    logits2,
    student_prob,
    epoch,
    epochs,
    threshold_start=0.80,
    threshold_end=0.65,
    pair_delta=0.05,
    pair_weight=0.7,
):
    """Refine target pseudo labels with EMA teacher multi-view agreement."""

    prob0_all = F.softmax(logits0, dim=1)
    prob1_all = F.softmax(logits1, dim=1)
    prob2_all = F.softmax(logits2, dim=1)

    teacher_prob_mean = (prob0_all + prob1_all + prob2_all) / 3.0

    conf0, label0 = prob0_all.max(dim=1)
    conf1, label1 = prob1_all.max(dim=1)
    conf2, label2 = prob2_all.max(dim=1)

    progress = min(max((epoch - 1) / max(1, epochs - 1), 0.0), 1.0)
    threshold = threshold_start - (threshold_start - threshold_end) * progress
    pair_threshold = max(0.0, threshold - pair_delta)

    student_label = student_prob.argmax(dim=1)

    refined_label = student_label.clone()
    refined_prob = student_prob.clone()
    sample_weight = torch.zeros_like(conf0)
    reliable_mask = torch.zeros_like(student_label, dtype=torch.bool)

    triple_mask = (
        (label0 == label1)
        & (label0 == label2)
        & (torch.minimum(torch.minimum(conf0, conf1), conf2) >= threshold)
    )

    refined_label[triple_mask] = label0[triple_mask]
    refined_prob[triple_mask] = teacher_prob_mean[triple_mask]
    sample_weight[triple_mask] = (
        conf0[triple_mask] + conf1[triple_mask] + conf2[triple_mask]
    ) / 3.0
    reliable_mask[triple_mask] = True

    pair01 = (
        (label0 == label1)
        & (~triple_mask)
        & (torch.minimum(conf0, conf1) >= pair_threshold)
    )
    pair02 = (
        (label0 == label2)
        & (~triple_mask)
        & (torch.minimum(conf0, conf2) >= pair_threshold)
    )
    pair12 = (
        (label1 == label2)
        & (~triple_mask)
        & (torch.minimum(conf1, conf2) >= pair_threshold)
    )

    fill = pair01 & (~reliable_mask)
    refined_label[fill] = label0[fill]
    refined_prob[fill] = teacher_prob_mean[fill]
    sample_weight[fill] = pair_weight * (conf0[fill] + conf1[fill]) / 2.0
    reliable_mask[fill] = True

    fill = pair02 & (~reliable_mask)
    refined_label[fill] = label0[fill]
    refined_prob[fill] = teacher_prob_mean[fill]
    sample_weight[fill] = pair_weight * (conf0[fill] + conf2[fill]) / 2.0
    reliable_mask[fill] = True

    fill = pair12 & (~reliable_mask)
    refined_label[fill] = label1[fill]
    refined_prob[fill] = teacher_prob_mean[fill]
    sample_weight[fill] = pair_weight * (conf1[fill] + conf2[fill]) / 2.0
    reliable_mask[fill] = True

    reliable_num = int(reliable_mask.sum().item())
    pair_num = int((reliable_mask & (~triple_mask)).sum().item())
    if reliable_num > 0:
        conf_mean = float(sample_weight[reliable_mask].mean().item())
        conf_max = float(sample_weight[reliable_mask].max().item())
    else:
        conf_mean = 0.0
        conf_max = 0.0

    num_classes = student_prob.size(1)
    teacher_label = teacher_prob_mean.argmax(dim=1)
    stats = {
        "threshold": float(threshold),
        "pair_threshold": float(pair_threshold),
        "triple_num": int(triple_mask.sum().item()),
        "pair_num": pair_num,
        "reliable_num": reliable_num,
        "reliable_ratio": float(reliable_mask.float().mean().item()),
        "teacher_hist": torch.bincount(
            teacher_label.detach().cpu(), minlength=num_classes
        ),
        "refined_hist": torch.bincount(
            refined_label.detach().cpu(), minlength=num_classes
        ),
        "student_hist": torch.bincount(
            student_label.detach().cpu(), minlength=num_classes
        ),
        "conf_mean": conf_mean,
        "conf_max": conf_max,
    }

    return refined_label, refined_prob, reliable_mask, sample_weight, stats
