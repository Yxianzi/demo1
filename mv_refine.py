"""Multi-view pseudo-label refinement for MLUDA."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _entropy_score(prob, eps=1e-6):
    num_classes = prob.size(1)
    entropy = -torch.sum(prob * torch.log(prob.clamp_min(eps)), dim=1)
    return (1.0 - entropy / math.log(max(2, num_classes))).clamp(min=0.0, max=1.0)


def _top1_margin(prob):
    if prob.size(1) < 2:
        return prob.new_zeros(prob.size(0))
    top2 = torch.topk(prob, k=2, dim=1).values
    return (top2[:, 0] - top2[:, 1]).clamp_min(0.0)


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
    reliability_min_score=0.35,
    topk_ratio_start=0.35,
    topk_ratio_end=0.75,
    min_per_class=1,
    max_top_ratio=0.45,
):
    """Select reliable class-balanced target pseudo labels from multi-view evidence.

    The EMA teacher acts as a reliability judge. It only replaces a student
    label when it is clearly stronger by confidence, margin, and entropy score.
    """

    prob0_all = F.softmax(logits0, dim=1)
    prob1_all = F.softmax(logits1, dim=1)
    prob2_all = F.softmax(logits2, dim=1)
    teacher_prob_mean = (prob0_all + prob1_all + prob2_all) / 3.0

    conf0, label0 = prob0_all.max(dim=1)
    conf1, label1 = prob1_all.max(dim=1)
    conf2, label2 = prob2_all.max(dim=1)

    teacher_conf, teacher_label = teacher_prob_mean.max(dim=1)
    teacher_entropy_score = _entropy_score(teacher_prob_mean)
    teacher_margin = _top1_margin(teacher_prob_mean)
    teacher_margin_score = teacher_margin.clamp_min(0.0)

    student_conf, student_label = student_prob.max(dim=1)
    student_entropy_score = _entropy_score(student_prob)
    student_margin = _top1_margin(student_prob)

    triple_agree = (label0 == label1) & (label0 == label2)
    pair_agree = (label0 == label1) | (label0 == label2) | (label1 == label2)

    view_agree_score = teacher_conf.new_zeros(teacher_conf.shape)
    view_agree_score[pair_agree] = float(pair_weight)
    view_agree_score[triple_agree] = 1.0

    agree_with_student = teacher_label == student_label
    teacher_stronger = (
        (~agree_with_student)
        & (teacher_conf >= student_conf + float(disagree_margin))
        & (teacher_margin >= student_margin + 0.03)
        & (teacher_entropy_score >= student_entropy_score)
    )

    student_agree_score = teacher_conf.new_zeros(teacher_conf.shape)
    student_agree_score[agree_with_student] = 1.0
    student_agree_score[teacher_stronger] = 0.75

    reliability_score = (
        teacher_conf
        * teacher_entropy_score
        * (1.0 + teacher_margin_score)
        * view_agree_score
        * student_agree_score
    )
    reliability_score = torch.where(
        torch.isfinite(reliability_score),
        reliability_score,
        torch.zeros_like(reliability_score),
    )

    candidate_label = student_label.clone()
    candidate_prob = student_prob.clone()

    candidate_prob[agree_with_student] = (
        0.5 * student_prob[agree_with_student]
        + 0.5 * teacher_prob_mean[agree_with_student]
    )
    candidate_label[teacher_stronger] = teacher_label[teacher_stronger]
    candidate_prob[teacher_stronger] = teacher_prob_mean[teacher_stronger]

    candidate_mask = (
        (view_agree_score > 0)
        & (student_agree_score > 0)
        & (reliability_score >= float(reliability_min_score))
    )

    batch_size = student_label.numel()
    num_classes = student_prob.size(1)
    progress = min(max((epoch - 1) / max(1, epochs - 1), 0.0), 1.0)
    threshold = threshold_start - (threshold_start - threshold_end) * progress
    pair_threshold = max(0.0, threshold - pair_delta)
    topk_ratio = topk_ratio_start + (topk_ratio_end - topk_ratio_start) * progress
    per_class_k = max(
        int(min_per_class),
        int(math.ceil(topk_ratio * batch_size / max(1, num_classes))),
    )

    accepted_mask = torch.zeros_like(candidate_mask)
    for class_id in range(num_classes):
        class_mask = candidate_mask & (candidate_label == class_id)
        idx = torch.nonzero(class_mask, as_tuple=False).view(-1)
        if idx.numel() == 0:
            continue
        order = torch.argsort(reliability_score[idx], descending=True)
        keep_idx = idx[order[:per_class_k]]
        accepted_mask[keep_idx] = True

    accepted_hist_device = torch.bincount(
        candidate_label[accepted_mask],
        minlength=num_classes,
    )
    accepted_total = int(accepted_hist_device.sum().item())
    accepted_top_ratio = (
        float((accepted_hist_device.max().float() / accepted_hist_device.sum().float().clamp_min(1.0)).item())
        if accepted_total > 0
        else 0.0
    )
    guard_reject = bool(accepted_total > 0 and accepted_top_ratio > float(max_top_ratio))

    if guard_enabled and guard_reject:
        max_class = int(torch.argmax(accepted_hist_device).item())
        max_class_idx = torch.nonzero(
            accepted_mask & (candidate_label == max_class),
            as_tuple=False,
        ).view(-1)
        other_count = accepted_total - int(max_class_idx.numel())
        if max_class_idx.numel() > 0:
            if other_count > 0:
                allowed = int(math.floor(float(max_top_ratio) * other_count / max(1e-6, 1.0 - float(max_top_ratio))))
                allowed = max(1, min(allowed, int(max_class_idx.numel())))
            else:
                allowed = 1

            if allowed < max_class_idx.numel():
                order = torch.argsort(reliability_score[max_class_idx], descending=True)
                drop_idx = max_class_idx[order[allowed:]]
                accepted_mask[drop_idx] = False

            accepted_hist_device = torch.bincount(
                candidate_label[accepted_mask],
                minlength=num_classes,
            )
            accepted_total = int(accepted_hist_device.sum().item())
            accepted_top_ratio = (
                float((accepted_hist_device.max().float() / accepted_hist_device.sum().float().clamp_min(1.0)).item())
                if accepted_total > 0
                else 0.0
            )

    refined_label = student_label.clone()
    refined_prob = student_prob.clone()
    sample_weight = torch.zeros_like(teacher_conf)
    reliable_mask = accepted_mask

    refined_label[accepted_mask] = candidate_label[accepted_mask]
    refined_prob[accepted_mask] = candidate_prob[accepted_mask]
    sample_weight[accepted_mask] = reliability_score[accepted_mask]

    student_hist_device = torch.bincount(student_label, minlength=num_classes)
    refined_hist_device = torch.bincount(refined_label, minlength=num_classes)

    student_nonzero = int((student_hist_device > 0).sum().item())
    refined_nonzero = int((refined_hist_device > 0).sum().item())
    accepted_nonzero = int((accepted_hist_device > 0).sum().item())

    student_total = student_hist_device.sum().float().clamp_min(1.0)
    refined_total = refined_hist_device.sum().float().clamp_min(1.0)

    student_top_ratio = float((student_hist_device.max().float() / student_total).item())
    refined_top_ratio = float((refined_hist_device.max().float() / refined_total).item())

    reliable_num = int(reliable_mask.sum().item())
    if reliable_num > 0:
        accepted_weight_mean = float(sample_weight[reliable_mask].mean().item())
        accepted_weight_max = float(sample_weight[reliable_mask].max().item())
    else:
        accepted_weight_mean = 0.0
        accepted_weight_max = 0.0

    teacher_student_agree_num = int((agree_with_student & candidate_mask).sum().item())
    teacher_stronger_num = int((teacher_stronger & candidate_mask).sum().item())
    rejected_disagree_num = int(((~agree_with_student) & (~teacher_stronger)).sum().item())

    pair_accepted = reliable_mask & pair_agree & (~triple_agree)
    stats = {
        "threshold": float(threshold),
        "pair_threshold": float(pair_threshold),
        "triple_num": int((reliable_mask & triple_agree).sum().item()),
        "pair_num": int(pair_accepted.sum().item()),
        "reliable_num": reliable_num,
        "reliable_ratio": float(reliable_mask.float().mean().item()),
        "teacher_hist": torch.bincount(
            teacher_label.detach().cpu(),
            minlength=num_classes,
        ),
        "refined_hist": refined_hist_device.detach().cpu(),
        "student_hist": student_hist_device.detach().cpu(),
        "conf_mean": accepted_weight_mean,
        "conf_max": accepted_weight_max,
        "guard_reject": guard_reject,
        "student_nonzero": student_nonzero,
        "refined_nonzero": refined_nonzero,
        "student_top_ratio": student_top_ratio,
        "refined_top_ratio": refined_top_ratio,
        "accepted_num": reliable_num,
        "class_cap": per_class_k,
        "accepted_hist": accepted_hist_device.detach().cpu(),
        "accepted_nonzero": accepted_nonzero,
        "accepted_top_ratio": accepted_top_ratio,
        "accepted_weight_mean": accepted_weight_mean,
        "accepted_weight_max": accepted_weight_max,
        "candidate_num": int(candidate_mask.sum().item()),
        "candidate_ratio": float(candidate_mask.float().mean().item()),
        "per_class_k": per_class_k,
        "topk_ratio": float(topk_ratio),
        "teacher_student_agree_num": teacher_student_agree_num,
        "teacher_stronger_num": teacher_stronger_num,
        "rejected_disagree_num": rejected_disagree_num,
    }

    return refined_label, refined_prob, reliable_mask, sample_weight, stats
