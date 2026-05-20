"""Multi-view pseudo-label refinement for MLUDA."""

from __future__ import annotations

import math

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
    guard_enabled=True,
    guard_min_class_drop=1,
    guard_top_ratio_delta=0.20,
    guard_max_top_ratio=0.60,
    class_cap_factor=2.0,
    disagree_margin=0.10,
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

    student_conf, student_label = student_prob.max(dim=1)

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

    teacher_label_candidate = refined_label.clone()
    teacher_prob_candidate = refined_prob.clone()
    teacher_weight_candidate = sample_weight.clone()
    teacher_reliable_candidate = reliable_mask.clone()

    agree_with_student = teacher_label_candidate == student_label

    teacher_stronger = (
        teacher_reliable_candidate
        & (~agree_with_student)
        & (teacher_weight_candidate >= student_conf + disagree_margin)
    )

    safe_candidate_mask = (
        teacher_reliable_candidate
        & (agree_with_student | teacher_stronger)
    )

    batch_size = student_label.numel()
    num_classes = student_prob.size(1)
    class_cap = max(1, int(math.ceil(class_cap_factor * batch_size / num_classes)))

    accepted_mask = torch.zeros_like(safe_candidate_mask)

    for c in range(num_classes):
        class_mask = safe_candidate_mask & (teacher_label_candidate == c)
        idx = torch.nonzero(class_mask, as_tuple=False).view(-1)
        if idx.numel() == 0:
            continue

        agree_idx = idx[agree_with_student[idx]]
        disagree_idx = idx[~agree_with_student[idx]]

        keep_list = []

        if agree_idx.numel() > 0:
            agree_scores = teacher_weight_candidate[agree_idx]
            order = torch.argsort(agree_scores, descending=True)
            agree_idx = agree_idx[order]
            keep_list.append(agree_idx)

        if disagree_idx.numel() > 0:
            disagree_scores = teacher_weight_candidate[disagree_idx]
            order = torch.argsort(disagree_scores, descending=True)
            disagree_idx = disagree_idx[order]
            keep_list.append(disagree_idx)

        if len(keep_list) > 0:
            keep_idx = torch.cat(keep_list, dim=0)[:class_cap]
            accepted_mask[keep_idx] = True

    final_refined_label = student_label.clone()
    final_refined_prob = student_prob.clone()
    final_sample_weight = torch.zeros_like(sample_weight)
    final_reliable_mask = accepted_mask

    final_refined_label[accepted_mask] = teacher_label_candidate[accepted_mask]
    final_refined_prob[accepted_mask] = teacher_prob_candidate[accepted_mask]
    final_sample_weight[accepted_mask] = teacher_weight_candidate[accepted_mask]

    refined_label = final_refined_label
    refined_prob = final_refined_prob
    sample_weight = final_sample_weight
    reliable_mask = final_reliable_mask

    student_hist_device = torch.bincount(student_label, minlength=num_classes)
    refined_hist_device = torch.bincount(refined_label, minlength=num_classes)

    student_nonzero = int((student_hist_device > 0).sum().item())
    refined_nonzero = int((refined_hist_device > 0).sum().item())

    student_total = student_hist_device.sum().float().clamp_min(1.0)
    refined_total = refined_hist_device.sum().float().clamp_min(1.0)

    student_top_ratio = float((student_hist_device.max().float() / student_total).item())
    refined_top_ratio = float((refined_hist_device.max().float() / refined_total).item())

    guard_reject = False

    if guard_enabled:
        guard_reject = (
            (refined_nonzero < student_nonzero - guard_min_class_drop)
            or (refined_top_ratio > student_top_ratio + guard_top_ratio_delta)
            or (refined_top_ratio > guard_max_top_ratio)
        )

    if guard_reject:
        refined_label = student_label.clone()
        refined_prob = student_prob.clone()
        reliable_mask = torch.zeros_like(student_label, dtype=torch.bool)
        sample_weight = torch.zeros_like(sample_weight)
        refined_hist_device = student_hist_device.clone()
        refined_nonzero = student_nonzero
        refined_top_ratio = student_top_ratio

    reliable_num = int(reliable_mask.sum().item())
    pair_num = int((reliable_mask & (~triple_mask)).sum().item())
    if reliable_num > 0:
        conf_mean = float(sample_weight[reliable_mask].mean().item())
        conf_max = float(sample_weight[reliable_mask].max().item())
    else:
        conf_mean = 0.0
        conf_max = 0.0

    teacher_label = teacher_prob_mean.argmax(dim=1)
    stats = {
        "threshold": float(threshold),
        "pair_threshold": float(pair_threshold),
        "triple_num": int((reliable_mask & triple_mask).sum().item()),
        "pair_num": pair_num,
        "reliable_num": reliable_num,
        "reliable_ratio": float(reliable_mask.float().mean().item()),
        "teacher_hist": torch.bincount(
            teacher_label.detach().cpu(), minlength=num_classes
        ),
        "refined_hist": refined_hist_device.detach().cpu(),
        "student_hist": student_hist_device.detach().cpu(),
        "conf_mean": conf_mean,
        "conf_max": conf_max,
        "guard_reject": bool(guard_reject),
        "student_nonzero": student_nonzero,
        "refined_nonzero": refined_nonzero,
        "student_top_ratio": student_top_ratio,
        "refined_top_ratio": refined_top_ratio,
        "accepted_num": int(reliable_mask.sum().item()),
        "class_cap": class_cap,
    }

    return refined_label, refined_prob, reliable_mask, sample_weight, stats
