"""Utilities for reliable pseudo labels in RP-MLUDA."""

from typing import Tuple

import torch
from torch import Tensor
from torch import nn


@torch.no_grad()
def update_ema_teacher(
    student_model: nn.Module,
    teacher_model: nn.Module,
    decay: float = 0.999,
) -> None:
    """Update teacher parameters and buffers with EMA from the student model.

    Floating-point tensors are updated as:
        teacher = decay * teacher + (1 - decay) * student

    Non-floating buffers, such as BatchNorm ``num_batches_tracked``, are copied
    directly from the student to keep module state valid.
    """
    student_state = student_model.state_dict()
    teacher_state = teacher_model.state_dict()

    for name, teacher_value in teacher_state.items():
        student_value = student_state[name].detach()
        if torch.is_floating_point(teacher_value):
            teacher_value.mul_(decay).add_(student_value.to(teacher_value.device), alpha=1.0 - decay)
        else:
            teacher_value.copy_(student_value.to(teacher_value.device))


def entropy_from_probs(probs: Tensor, eps: float = 1e-8) -> Tensor:
    """Compute per-sample entropy from softmax probabilities.

    Args:
        probs: Probability tensor with shape ``[B, C]``.
        eps: Small value for numerical stability.

    Returns:
        Entropy tensor with shape ``[B]``.
    """
    return -(probs * torch.log(probs.clamp_min(eps))).sum(dim=1)


def dynamic_threshold(
    epoch: int,
    epochs: int,
    start: float = 0.70,
    end: float = 0.95,
) -> float:
    """Linearly increase the confidence threshold from ``start`` to ``end``.

    ``epoch`` is expected to start from 1.
    """
    if epochs <= 1:
        return end

    progress = (epoch - 1) / float(epochs - 1)
    progress = min(max(progress, 0.0), 1.0)
    return start + (end - start) * progress


def get_reliable_pseudo_labels(
    logits: Tensor,
    epoch: int,
    epochs: int,
    threshold_start: float = 0.70,
    threshold_end: float = 0.95,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Generate pseudo labels and a confidence-based reliable mask.

    Args:
        logits: Model logits with shape ``[B, C]``.
        epoch: Current epoch, starting from 1.
        epochs: Total number of training epochs.
        threshold_start: Initial confidence threshold.
        threshold_end: Final confidence threshold.

    Returns:
        ``conf``, ``pseudo_label``, ``reliable_mask``, and ``probs`` tensors on
        the same device as ``logits``.
    """
    probs = torch.softmax(logits, dim=1)
    conf, pseudo_label = torch.max(probs, dim=1)
    threshold = dynamic_threshold(epoch, epochs, threshold_start, threshold_end)
    reliable_mask = conf >= threshold
    return conf, pseudo_label, reliable_mask, probs
