"""Train SR-SSFTT on the Houston cross-scene HSI domain adaptation task.

SR-SSFTT: Spatial-Reliability Guided Spectral-Spatial Feature Tokenization
Transformer.

Target labels are kept in the target DataLoader only for evaluation and debug
printing. They are never used by the training losses in this script.
"""

import argparse
import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.data import DataLoader, TensorDataset

import mmd
import utils
from UtilsCMS import ILDA
from config_Houston import pca_n, radius, seeds
from losses.prototype_alignment import spatial_reliable_prototype_alignment
from losses.spatial_reliability import (
    boundary_preserving_spatial_consistency,
    build_target_coordinates,
    compute_spatial_reliability,
    spatial_sort_indices,
)
from models.sr_ssftt import SRSSFTT
from prototype_memory import PrototypeMemory
from rp_utils import update_ema_teacher


def parse_value(value):
    value = value.strip()
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith(("'", '"')) and value.endswith(("'", '"')):
        return value[1:-1]
    try:
        if "." in value or "e" in value.lower():
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_config(path):
    """Load a flat YAML config without making PyYAML a hard dependency."""

    try:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        config = {}
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.split("#", 1)[0].strip()
                if not line or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                config[key.strip()] = parse_value(value)
        return config


def numpy_to_tensor(data, numpy_dtype, torch_dtype):
    array = np.ascontiguousarray(data, dtype=numpy_dtype)
    # Use frombuffer instead of from_numpy because some local PyTorch/NumPy
    # combinations raise "expected np.ndarray (got numpy.ndarray)".
    return torch.frombuffer(array, dtype=torch_dtype).reshape(array.shape)


def dynamic_threshold(epoch, epochs, start, end):
    if epochs <= 1:
        return float(end)
    progress = min(max((epoch - 1) / float(epochs - 1), 0.0), 1.0)
    return float(start + (end - start) * progress)


def sigmoid_rampup(epoch, warmup_epochs, total_epochs):
    if epoch <= warmup_epochs:
        return 0.0
    length = max(1, total_epochs - warmup_epochs)
    progress = min(max((epoch - warmup_epochs) / float(length), 0.0), 1.0)
    return float(math.exp(-5.0 * (1.0 - progress) ** 2))


def minimum_class_confusion_loss(logits, temperature=2.5):
    """MCC loss on target logits."""

    probs = F.softmax(logits / temperature, dim=1)
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=1)
    weights = 1.0 + torch.exp(-entropy)
    weights = logits.size(0) * weights / weights.sum().clamp_min(1e-8)
    class_confusion = torch.matmul((probs * weights.view(-1, 1)).t(), probs)
    class_confusion = class_confusion / class_confusion.sum(dim=1, keepdim=True).clamp_min(1e-8)
    num_classes = logits.size(1)
    off_diag = class_confusion.sum() - torch.trace(class_confusion)
    return off_diag / num_classes


def target_diversity_loss(target_logits):
    probs = F.softmax(target_logits, dim=1)
    mean_probs = probs.mean(dim=0)
    uniform = torch.full_like(mean_probs, 1.0 / mean_probs.numel())
    return F.kl_div(mean_probs.clamp_min(1e-8).log(), uniform, reduction="sum")


def build_patch_reliability_map(target_x, reliability, sigma=1.0):
    b, c, h, w = target_x.shape
    center = target_x[:, :, h // 2, w // 2].view(b, c, 1, 1)
    dist = ((target_x - center) ** 2).mean(dim=1)
    safe_sigma = max(float(sigma), 1e-6)
    sim = torch.exp(-dist / (2 * safe_sigma**2))
    reliability = reliability.to(device=target_x.device, dtype=target_x.dtype).view(b, 1, 1)
    return reliability * sim


def spatial_voting_refinement(base_probs, affinity, alpha):
    """Blend sample probabilities with spatial-spectral neighbor votes."""

    row_sum = affinity.sum(dim=1, keepdim=True)
    neighbor_probs = torch.matmul(affinity, base_probs) / row_sum.clamp_min(1e-8)
    neighbor_probs = torch.where(row_sum > 0, neighbor_probs, base_probs)
    refined = (1.0 - alpha) * base_probs + alpha * neighbor_probs
    return refined / refined.sum(dim=1, keepdim=True).clamp_min(1e-8)


def class_balanced_reliable_selection(labels, scores, min_threshold=0.45, topk_per_class=8, num_classes=None):
    """Select top-k reliable target samples independently within each pseudo class."""

    labels = labels.to(dtype=torch.long).view(-1)
    scores = scores.to(device=labels.device).view(-1)
    if num_classes is None:
        num_classes = int(labels.max().item()) + 1 if labels.numel() > 0 else 0

    selected = torch.zeros_like(labels, dtype=torch.bool)
    selected_per_class = torch.zeros(int(num_classes), device=labels.device, dtype=torch.long)
    if int(topk_per_class) <= 0:
        return selected, selected_per_class

    for class_id in range(num_classes):
        class_mask = labels == class_id
        index = torch.nonzero(class_mask, as_tuple=False).view(-1)
        if index.numel() == 0:
            continue
        class_scores = scores[index]
        keep = min(int(topk_per_class), index.numel())
        order = torch.argsort(class_scores, descending=True)
        top_index = index[order[:keep]]
        top_index = top_index[scores[top_index] >= float(min_threshold)]
        if top_index.numel() == 0:
            continue
        selected[top_index] = True
        selected_per_class[class_id] = top_index.numel()
    return selected, selected_per_class


def weighted_kl_loss(student_logits, teacher_probs, weight=None, mask=None):
    teacher_probs = teacher_probs.detach()
    per_sample = F.kl_div(
        F.log_softmax(student_logits, dim=1),
        teacher_probs,
        reduction="none",
    ).sum(dim=1)

    if mask is not None:
        mask = mask.to(student_logits.device, dtype=torch.bool)
        per_sample = per_sample[mask]
        if weight is not None:
            weight = weight.to(student_logits.device, student_logits.dtype)[mask]
    elif weight is not None:
        weight = weight.to(student_logits.device, student_logits.dtype)

    if per_sample.numel() == 0:
        return student_logits.new_tensor(0.0)

    if weight is None:
        return per_sample.mean()

    return (per_sample * weight).sum() / weight.sum().clamp_min(1e-8)


@torch.no_grad()
def update_reliability_weighted_memory(
    memory,
    features,
    labels,
    reliability,
    mask,
    min_count_per_class=1,
):
    """EMA-update target prototype memory with reliability-weighted means."""

    device = memory.prototypes.device
    update_counts = torch.zeros(memory.num_classes, device=device, dtype=torch.long)
    if mask is None:
        return update_counts
    dtype = memory.prototypes.dtype
    features = features.detach().to(device=device, dtype=dtype)
    labels = labels.detach().to(device=device, dtype=torch.long).view(-1)
    reliability = reliability.detach().to(device=device, dtype=dtype).view(-1)
    mask = mask.detach().to(device=device, dtype=torch.bool).view(-1)

    valid = mask & (labels >= 0) & (labels < memory.num_classes)
    if not valid.any():
        return update_counts

    for class_id in range(memory.num_classes):
        class_mask = valid & (labels == class_id)
        class_count = int(class_mask.sum().item())
        if class_count < int(min_count_per_class):
            continue

        weights = reliability[class_mask].clamp_min(0.0)
        weight_sum = weights.sum()
        if weight_sum.item() <= memory.eps:
            continue

        class_mean = (features[class_mask] * weights.view(-1, 1)).sum(dim=0) / weight_sum.clamp_min(memory.eps)
        if not memory.initialized[class_id]:
            updated = class_mean
            memory.initialized[class_id] = True
        else:
            updated = memory.momentum * memory.prototypes[class_id] + (1.0 - memory.momentum) * class_mean

        memory.prototypes[class_id] = F.normalize(updated.unsqueeze(0), p=2, dim=1, eps=memory.eps).squeeze(0)
        update_counts[class_id] = class_count

    return update_counts


@torch.no_grad()
def get_fused_memory_prototypes(
    source_memory,
    target_memory,
    target_counts=None,
    proto_count_tau=20,
    use_target=True,
):
    """Build fused prototypes from EMA source/target memories for train/eval."""

    source_proto = source_memory.get()
    source_valid = source_memory.initialized
    if not use_target or target_memory is None:
        if not source_valid.any():
            return None
        fused = source_proto.clone()
        fused[~source_valid] = 0.0
        return F.normalize(fused, p=2, dim=1, eps=1e-8)

    target_proto = target_memory.get().to(device=source_proto.device, dtype=source_proto.dtype)
    target_valid = target_memory.initialized.to(source_valid.device)
    valid = source_valid | target_valid
    if not valid.any():
        return None

    fused = source_proto.clone()
    both_valid = source_valid & target_valid
    if target_counts is None:
        target_counts = torch.zeros(source_memory.num_classes, device=source_proto.device, dtype=source_proto.dtype)
    else:
        target_counts = target_counts.to(device=source_proto.device, dtype=source_proto.dtype).view(-1)
    safe_tau = max(float(proto_count_tau), 1e-6)
    target_ratio = target_counts / (target_counts + safe_tau)
    target_only = (~source_valid) & target_valid & (target_counts > 0)

    if both_valid.any():
        ratio = target_ratio[both_valid].view(-1, 1)
        fused[both_valid] = (1.0 - ratio) * source_proto[both_valid] + ratio * target_proto[both_valid]
    if target_only.any():
        fused[target_only] = target_proto[target_only]
    fused[~valid] = 0.0
    return F.normalize(fused, p=2, dim=1, eps=1e-8)


def compute_lmmd_loss(source_features, target_features, source_labels, target_probs, target_weight, batch_size, num_classes):
    """Use reliability-weighted LMMD when available, otherwise fall back."""

    if hasattr(mmd, "weighted_lmmd"):
        return mmd.weighted_lmmd(
            source_features,
            target_features,
            source_labels,
            target_probs.detach(),
            t_weight=target_weight.detach(),
            BATCH_SIZE=batch_size,
            CLASS_NUM=num_classes,
        )

    return mmd.lmmd(
        source_features,
        target_features,
        source_labels,
        target_probs.detach(),
        BATCH_SIZE=batch_size,
        CLASS_NUM=num_classes,
    )


def evaluate_target_domain(
    model,
    test_loader,
    device,
    num_classes,
    prototypes=None,
    use_reliability_eval=True,
    spatial_k=6,
    sigma_spatial=2.5,
    sigma_spectral=1.0,
    reliability_map_sigma=1.0,
):
    model.eval()
    predict = np.array([], dtype=np.int64)
    labels = np.array([], dtype=np.int64)
    total_hit = 0

    with torch.no_grad():
        for batch in test_loader:
            coords = None
            if len(batch) == 3:
                data, label, coords = batch
            else:
                data, label = batch
            data = data.to(device)
            if use_reliability_eval and coords is not None:
                coords = coords.to(device)
                pre_out = model(data, reliability=None, domain="target", prototypes=prototypes)
                probs = F.softmax(pre_out["logits"], dim=1)
                reliability, _affinity, _spatial_stats = compute_spatial_reliability(
                    probs.detach(),
                    coords,
                    data,
                    k=int(spatial_k),
                    sigma_spatial=float(sigma_spatial),
                    sigma_spectral=float(sigma_spectral),
                )
                reliability = reliability.detach()
                reliability_map = build_patch_reliability_map(
                    data,
                    reliability,
                    sigma=float(reliability_map_sigma),
                )
                out = model(
                    data,
                    reliability=reliability,
                    reliability_map=reliability_map,
                    domain="target",
                    prototypes=prototypes,
                )
            else:
                out = model(data, reliability=None, domain="target", prototypes=prototypes)
            pred = out["logits"].argmax(dim=1)
            pred_np = np.asarray(pred.detach().cpu().tolist(), dtype=np.int64)
            label_np = np.asarray(label.cpu().tolist(), dtype=np.int64)
            total_hit += int((pred_np == label_np).sum())
            predict = np.append(predict, pred_np)
            labels = np.append(labels, label_np)

    class_ids = np.arange(num_classes)
    confusion = metrics.confusion_matrix(labels, predict, labels=class_ids)
    class_totals = np.sum(confusion, axis=1, dtype=np.float64)
    class_accuracy = np.divide(
        np.diag(confusion),
        class_totals,
        out=np.zeros(num_classes, dtype=np.float64),
        where=class_totals != 0,
    )
    oa = 100.0 * total_hit / max(1, len(test_loader.dataset))
    aa = float(np.mean(class_accuracy))
    kappa = float(metrics.cohen_kappa_score(labels, predict, labels=class_ids))
    return {
        "oa": oa,
        "aa": aa,
        "kappa": kappa,
        "class_accuracy": class_accuracy,
        "predict": predict,
        "labels": labels,
        "total_hit": total_hit,
        "total_count": len(test_loader.dataset),
    }


def print_eval_result(result):
    print("\tOA: {}/{} ({:.2f}%)".format(result["total_hit"], result["total_count"], result["oa"]))
    print("\tAA: {:.2f}%".format(100.0 * result["aa"]))
    print("\tKappa: {:.4f}".format(100.0 * result["kappa"]))
    print("\taccuracy for each class:")
    for class_id, acc in enumerate(result["class_accuracy"]):
        print("\tClass {}: {:.2f}".format(class_id, 100.0 * acc))


def print_average_result(acc_values, class_acc_values, kappa_values):
    aa = np.mean(class_acc_values, axis=1)
    print("average OA: {:.2f} +- {:.2f}".format(np.mean(acc_values), np.std(acc_values)))
    print("average AA: {:.2f} +- {:.2f}".format(100.0 * np.mean(aa), 100.0 * np.std(aa)))
    print("average kappa: {:.4f} +- {:.4f}".format(100.0 * np.mean(kappa_values), 100.0 * np.std(kappa_values)))
    print("accuracy for each class:")
    class_mean = np.mean(class_acc_values, axis=0)
    class_std = np.std(class_acc_values, axis=0)
    for class_id in range(class_acc_values.shape[1]):
        print("Class {}: {:.2f} +- {:.2f}".format(class_id, 100.0 * class_mean[class_id], 100.0 * class_std[class_id]))


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train SR-SSFTT on Houston cross-scene DA.")
    parser.add_argument("--config", default="configs/sr_ssftt_houston.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--n_datasets", type=int, default=None)
    parser.add_argument("--disable_reliability_tokenizer", action="store_true")
    parser.add_argument("--disable_domain_adapter", action="store_true")
    parser.add_argument("--disable_proto_head", action="store_true")
    parser.add_argument("--disable_spatial_reliability", action="store_true")
    parser.add_argument("--disable_reliability_eval", action="store_true")
    parser.add_argument("--disable_proto_alignment", action="store_true")
    parser.add_argument("--disable_spatial_consistency", action="store_true")
    parser.add_argument("--disable_pseudo_ce", action="store_true")
    return parser


def main():
    args = build_arg_parser().parse_args()
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.n_datasets is not None:
        cfg["n_datasets"] = args.n_datasets

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = int(cfg["num_classes"])
    n_band = int(cfg["n_band"])
    half_width = int(cfg["half_width"])
    patch_size = int(cfg["patch_size"])
    epochs = int(cfg["epochs"])
    batch_size = int(cfg["batch_size"])
    eval_interval = int(cfg.get("eval_interval", 10))
    n_datasets = min(int(cfg.get("n_datasets", 3)), len(seeds))
    use_reliability_eval = bool(cfg.get("use_reliability_eval", True)) and not args.disable_reliability_eval
    pseudo_warmup_epochs = int(cfg.get("pseudo_warmup_epochs", 15))
    target_proto_warmup_epochs = int(cfg.get("target_proto_warmup_epochs", 20))
    proto_head_warmup_epochs = int(cfg.get("proto_head_warmup_epochs", 10))

    data_s, label_s = utils.load_data_houston(cfg["source_data"], cfg["source_label"])
    data_t, label_t = utils.load_data_houston(cfg["target_data"], cfg["target_label"])
    data_s, data_t = ILDA(data_s, data_t, pca_n, radius)

    cross_entropy = nn.CrossEntropyLoss().to(device)
    acc_all = np.zeros((n_datasets, 1), dtype=np.float64)
    class_acc_all = np.zeros((n_datasets, num_classes), dtype=np.float64)
    kappa_all = np.zeros((n_datasets, 1), dtype=np.float64)
    best_acc_all = 0.0
    best_predict_all = []
    train_time = 0.0
    test_time = 0.0

    for i_dataset in range(n_datasets):
        print("#######################idataset######################## ", i_dataset)
        utils.set_seed(seeds[i_dataset])

        train_x, train_y = utils.get_sample_data(data_s, label_s, half_width, 180)
        test_id, test_x, test_y, _g, rand_perm, row, column = utils.get_all_data(data_t, label_t, half_width)

        target_coords = build_target_coordinates(row, column, rand_perm, half_width)
        sort_index = spatial_sort_indices(target_coords)
        sort_np = np.asarray(sort_index.cpu().tolist(), dtype=np.int64)
        test_x = test_x[sort_np]
        test_y = test_y[sort_np]
        target_coords = target_coords[sort_index]

        source_dataset = TensorDataset(
            numpy_to_tensor(train_x, np.float32, torch.float32),
            numpy_to_tensor(train_y, np.int64, torch.long),
        )
        target_dataset = TensorDataset(
            numpy_to_tensor(test_x, np.float32, torch.float32),
            numpy_to_tensor(test_y, np.int64, torch.long),
            target_coords.float(),
        )

        source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        target_train_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        target_test_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        model = SRSSFTT(
            n_band=n_band,
            num_classes=num_classes,
            patch_size=patch_size,
            num_tokens=int(cfg["num_tokens"]),
            token_dim=int(cfg["token_dim"]),
            transformer_depth=int(cfg["transformer_depth"]),
            num_heads=int(cfg["num_heads"]),
            adapter_bottleneck=int(cfg["adapter_bottleneck"]),
            proto_temperature=float(cfg["proto_temperature"]),
            proto_logit_weight=float(cfg["proto_logit_weight"]),
            use_reliability_tokenizer=not args.disable_reliability_tokenizer,
            use_domain_adapter=not args.disable_domain_adapter,
            use_proto_head=not args.disable_proto_head,
        ).to(device)
        teacher_model = copy.deepcopy(model).to(device)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False

        source_proto_memory = PrototypeMemory(
            num_classes,
            model.feature_dim,
            momentum=0.9,
        ).to(device)
        target_proto_memory = PrototypeMemory(
            num_classes,
            model.feature_dim,
            momentum=0.9,
        ).to(device)
        target_proto_counts = torch.zeros(num_classes, device=device, dtype=torch.long)

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(cfg["lr"]),
            momentum=float(cfg["momentum"]),
            weight_decay=float(cfg["weight_decay"]),
        )

        last_result = None
        train_start = time.time()
        target_iter = iter(target_train_loader)
        steps_per_epoch = len(source_loader)
        print(
            "Training SR-SSFTT... steps_per_epoch:{}, target_batches:{}".format(
                steps_per_epoch,
                len(target_train_loader),
            )
        )

        for epoch in range(1, epochs + 1):
            learning_rate = float(cfg["lr"]) / math.pow((1 + 10 * (epoch - 1) / epochs), 0.75)
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate

            pseudo_active = epoch > pseudo_warmup_epochs
            target_proto_active = epoch > target_proto_warmup_epochs
            proto_head_active = (epoch > proto_head_warmup_epochs) and not args.disable_proto_head

            model.train()
            teacher_model.eval()
            source_iter = iter(source_loader)

            meter = {
                "source_ce": 0.0,
                "lmmd": 0.0,
                "mcc": 0.0,
                "pseudo_ce": 0.0,
                "proto": 0.0,
                "sp_pred": 0.0,
                "sp_feat": 0.0,
                "cons": 0.0,
                "diversity": 0.0,
                "reliability_mean": 0.0,
                "reliability_max": 0.0,
                "selected_num": 0,
                "loss": 0.0,
            }
            selected_per_class = torch.zeros(num_classes, dtype=torch.long)
            target_prob_sum = torch.zeros(num_classes, dtype=torch.float64)
            target_prob_count = 0
            epoch_target_features = []
            epoch_target_labels = []
            epoch_target_reliability = []
            num_batches = 0
            threshold = dynamic_threshold(
                epoch,
                epochs,
                float(cfg["pseudo_threshold_start"]),
                float(cfg["pseudo_threshold_end"]),
            )
            ramp = sigmoid_rampup(epoch, int(cfg["warmup_epochs"]), epochs)

            for _step in range(steps_per_epoch):
                source_x, source_y = next(source_iter)
                try:
                    target_x, _target_y, target_coords_batch = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_train_loader)
                    target_x, _target_y, target_coords_batch = next(target_iter)

                source_x = source_x.to(device)
                source_y_cuda = source_y.to(device)
                target_x = target_x.to(device)
                target_coords_batch = target_coords_batch.to(device)
                if not proto_head_active:
                    prototypes_for_head = None
                else:
                    prototypes_for_head = get_fused_memory_prototypes(
                        source_proto_memory,
                        target_proto_memory,
                        target_counts=target_proto_counts,
                        proto_count_tau=float(cfg.get("proto_count_tau", 20)),
                        use_target=target_proto_active,
                    )

                source_out = model(
                    source_x,
                    reliability=None,
                    domain="source",
                    prototypes=prototypes_for_head,
                )
                target_pre_out = model(
                    target_x,
                    reliability=None,
                    domain="target",
                    prototypes=prototypes_for_head,
                )

                with torch.no_grad():
                    teacher_probs0 = F.softmax(
                        teacher_model(target_x, domain="target", prototypes=prototypes_for_head)["logits"],
                        dim=1,
                    )
                    target_aug1 = utils.radiation_noise(target_x)
                    target_aug2 = utils.flip_augmentation(target_x)
                    teacher_probs1 = F.softmax(
                        teacher_model(target_aug1, domain="target", prototypes=prototypes_for_head)["logits"],
                        dim=1,
                    )
                    teacher_probs2 = F.softmax(
                        teacher_model(target_aug2, domain="target", prototypes=prototypes_for_head)["logits"],
                        dim=1,
                    )
                    teacher_probs = (teacher_probs0 + teacher_probs1 + teacher_probs2) / 3.0

                pre_probs = F.softmax(target_pre_out["logits"].detach(), dim=1)
                if args.disable_spatial_reliability:
                    reliability = pre_probs.max(dim=1)[0]
                    affinity = pre_probs.new_zeros((pre_probs.size(0), pre_probs.size(0)))
                    spatial_stats = {
                        "reliability_mean": float(reliability.mean().item()),
                        "reliability_max": float(reliability.max().item()),
                        "edge_num": 0,
                    }
                else:
                    reliability, affinity, spatial_stats = compute_spatial_reliability(
                        pre_probs,
                        target_coords_batch,
                        target_x,
                        k=int(cfg["spatial_k"]),
                        sigma_spatial=float(cfg["sigma_spatial"]),
                        sigma_spectral=float(cfg["sigma_spectral"]),
                    )

                target_reliability_for_tokenizer = reliability.detach()
                target_reliability_map = build_patch_reliability_map(
                    target_x,
                    target_reliability_for_tokenizer,
                    sigma=float(cfg.get("reliability_map_sigma", 1.0)),
                )

                target_out = model(
                    target_x,
                    reliability=target_reliability_for_tokenizer,
                    reliability_map=target_reliability_map,
                    domain="target",
                    prototypes=prototypes_for_head,
                )
                source_features = source_out["features"]
                target_features = target_out["features"]
                source_logits = source_out["logits"]
                target_logits = target_out["logits"]
                target_probs = F.softmax(target_logits, dim=1)
                target_prob_sum += target_probs.detach().sum(dim=0).cpu().double()
                target_prob_count += int(target_probs.size(0))

                with torch.no_grad():
                    refinement_base = 0.5 * target_probs.detach() + 0.5 * teacher_probs.detach()
                    refined_probs = spatial_voting_refinement(
                        refinement_base,
                        affinity.detach(),
                        alpha=float(cfg["spatial_vote_alpha"]),
                    )
                    pseudo_labels = refined_probs.argmax(dim=1)
                    if pseudo_active:
                        selected_mask, batch_selected_per_class = class_balanced_reliable_selection(
                            pseudo_labels,
                            reliability.detach(),
                            min_threshold=threshold,
                            topk_per_class=int(cfg["topk_per_class"]),
                            num_classes=num_classes,
                        )
                    else:
                        selected_mask = torch.zeros_like(pseudo_labels, dtype=torch.bool)
                        batch_selected_per_class = torch.zeros(num_classes, device=device, dtype=torch.long)

                source_ce = cross_entropy(source_logits, source_y_cuda)
                lmmd_loss = compute_lmmd_loss(
                    source_features,
                    target_features,
                    source_y_cuda,
                    refined_probs.detach(),
                    reliability.detach(),
                    batch_size=batch_size,
                    num_classes=num_classes,
                )
                mcc_loss = minimum_class_confusion_loss(target_logits)
                diversity_loss = target_diversity_loss(target_logits)

                if selected_mask.any() and not args.disable_pseudo_ce:
                    pseudo_ce_loss = F.cross_entropy(target_logits[selected_mask], pseudo_labels[selected_mask])
                else:
                    pseudo_ce_loss = target_logits.new_tensor(0.0)

                if selected_mask.any() and not args.disable_proto_alignment:
                    proto_align_loss = spatial_reliable_prototype_alignment(
                        source_features,
                        source_y_cuda,
                        target_features,
                        pseudo_labels,
                        reliability,
                        num_classes,
                        target_mask=selected_mask,
                    )
                else:
                    proto_align_loss = target_logits.new_tensor(0.0)

                if not args.disable_spatial_consistency:
                    spatial_pred_loss, spatial_feat_loss, _consistency_stats = boundary_preserving_spatial_consistency(
                        target_features,
                        target_probs.detach(),
                        target_coords_batch,
                        target_x,
                        reliability.detach(),
                        k=int(cfg["spatial_k"]),
                        sigma_spatial=float(cfg["sigma_spatial"]),
                        sigma_spectral=float(cfg["sigma_spectral"]),
                    )
                else:
                    spatial_pred_loss = target_logits.new_tensor(0.0)
                    spatial_feat_loss = target_logits.new_tensor(0.0)

                if pseudo_active and selected_mask.any():
                    teacher_consistency_loss = weighted_kl_loss(
                        target_logits,
                        refined_probs.detach(),
                        weight=reliability.detach(),
                        mask=selected_mask,
                    )
                else:
                    teacher_consistency_loss = target_logits.new_tensor(0.0)

                loss_warm = (
                    source_ce
                    + float(cfg["lambda_lmmd"]) * lmmd_loss
                    + float(cfg["lambda_mcc"]) * mcc_loss
                )
                loss_extra = (
                    float(cfg["lambda_pseudo"]) * pseudo_ce_loss
                    + float(cfg["lambda_proto"]) * proto_align_loss
                    + float(cfg["lambda_sp_pred"]) * spatial_pred_loss
                    + float(cfg["lambda_sp_feat"]) * spatial_feat_loss
                    + float(cfg["lambda_cons"]) * teacher_consistency_loss
                )
                total_loss = loss_warm + ramp * loss_extra + float(cfg.get("lambda_div", 0.01)) * diversity_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                update_ema_teacher(model, teacher_model, decay=float(cfg["ema_decay"]))

                with torch.no_grad():
                    source_proto_memory.update(
                        source_features.detach(),
                        source_y_cuda.detach(),
                    )
                    if target_proto_active and selected_mask.any():
                        selected_detached = selected_mask.detach()
                        epoch_target_features.append(target_features.detach()[selected_detached])
                        epoch_target_labels.append(pseudo_labels.detach()[selected_detached])
                        epoch_target_reliability.append(reliability.detach()[selected_detached])

                selected_per_class += batch_selected_per_class.detach().cpu()
                meter["source_ce"] += float(source_ce.item())
                meter["lmmd"] += float(lmmd_loss.item())
                meter["mcc"] += float(mcc_loss.item())
                meter["pseudo_ce"] += float(pseudo_ce_loss.item())
                meter["proto"] += float(proto_align_loss.item())
                meter["sp_pred"] += float(spatial_pred_loss.item())
                meter["sp_feat"] += float(spatial_feat_loss.item())
                meter["cons"] += float(teacher_consistency_loss.item())
                meter["diversity"] += float(diversity_loss.item())
                meter["reliability_mean"] += float(spatial_stats["reliability_mean"])
                meter["reliability_max"] += float(spatial_stats["reliability_max"])
                meter["selected_num"] += int(selected_mask.sum().item())
                meter["loss"] += float(total_loss.item())
                num_batches += 1

            if epoch_target_features:
                epoch_features = torch.cat(epoch_target_features, dim=0)
                epoch_labels = torch.cat(epoch_target_labels, dim=0)
                epoch_reliability = torch.cat(epoch_target_reliability, dim=0)
                epoch_mask = torch.ones(epoch_labels.size(0), device=epoch_labels.device, dtype=torch.bool)
                target_proto_update_per_class = update_reliability_weighted_memory(
                    target_proto_memory,
                    epoch_features,
                    epoch_labels,
                    epoch_reliability,
                    epoch_mask,
                    min_count_per_class=int(cfg.get("min_proto_update_per_class", 3)),
                )
            else:
                target_proto_update_per_class = torch.zeros(num_classes, device=device, dtype=torch.long)

            target_proto_counts = selected_per_class.to(device=device)
            mean_target_probs = (target_prob_sum / max(1, target_prob_count)).tolist()
            mean_target_probs = [round(float(prob), 4) for prob in mean_target_probs]
            target_proto_update_list = target_proto_update_per_class.detach().cpu().tolist()
            target_proto_initialized = target_proto_memory.initialized.detach().cpu().tolist()
            denom = max(1, num_batches)
            print(
                "epoch {:>3d}: lr:{:.5f}, pseudo_active:{}, target_proto_active:{}, "
                "proto_head_active:{}, source_ce:{:6.4f}, lmmd_loss:{:6.4f}, "
                "mcc_loss:{:6.4f}, pseudo_ce_loss:{:6.4f}, proto_align_loss:{:6.4f}, "
                "spatial_pred_loss:{:6.4f}, spatial_feat_loss:{:6.4f}, "
                "teacher_consistency_loss:{:6.4f}, diversity_loss:{:6.4f}, "
                "reliability_mean:{:6.4f}, "
                "reliability_max:{:6.4f}, selected_pseudo_num:{:>4d}, "
                "mean_target_probs:{}, selected_per_class:{}, target_proto_update_per_class:{}, "
                "target_proto_initialized:{}, threshold:{:6.4f}, ramp:{:6.4f}, total_loss:{:6.4f}".format(
                    epoch,
                    learning_rate,
                    pseudo_active,
                    target_proto_active,
                    proto_head_active,
                    meter["source_ce"] / denom,
                    meter["lmmd"] / denom,
                    meter["mcc"] / denom,
                    meter["pseudo_ce"] / denom,
                    meter["proto"] / denom,
                    meter["sp_pred"] / denom,
                    meter["sp_feat"] / denom,
                    meter["cons"] / denom,
                    meter["diversity"] / denom,
                    meter["reliability_mean"] / denom,
                    meter["reliability_max"] / denom,
                    meter["selected_num"],
                    mean_target_probs,
                    selected_per_class.tolist(),
                    target_proto_update_list,
                    target_proto_initialized,
                    threshold,
                    ramp,
                    meter["loss"] / denom,
                )
            )

            if epoch % eval_interval == 0 or epoch == epochs:
                test_begin = time.time()
                print("Testing epoch {} ...".format(epoch))
                eval_prototypes = None
                if proto_head_active:
                    eval_prototypes = get_fused_memory_prototypes(
                        source_proto_memory,
                        target_proto_memory,
                        target_counts=target_proto_counts,
                        proto_count_tau=float(cfg.get("proto_count_tau", 20)),
                        use_target=target_proto_active,
                    )
                last_result = evaluate_target_domain(
                    model,
                    target_test_loader,
                    device,
                    num_classes,
                    prototypes=eval_prototypes,
                    use_reliability_eval=use_reliability_eval,
                    spatial_k=int(cfg["spatial_k"]),
                    sigma_spatial=float(cfg["sigma_spatial"]),
                    sigma_spectral=float(cfg["sigma_spectral"]),
                    reliability_map_sigma=float(cfg.get("reliability_map_sigma", 1.0)),
                )
                print_eval_result(last_result)
                test_time += time.time() - test_begin
                if last_result["oa"] > best_acc_all:
                    best_acc_all = last_result["oa"]
                    best_predict_all = last_result["predict"]

        train_time += time.time() - train_start
        if last_result is None:
            final_target_proto_active = epochs > target_proto_warmup_epochs
            final_proto_head_active = (epochs > proto_head_warmup_epochs) and not args.disable_proto_head
            last_result = evaluate_target_domain(
                model,
                target_test_loader,
                device,
                num_classes,
                prototypes=(
                    None
                    if not final_proto_head_active
                    else get_fused_memory_prototypes(
                        source_proto_memory,
                        target_proto_memory,
                        target_counts=target_proto_counts,
                        proto_count_tau=float(cfg.get("proto_count_tau", 20)),
                        use_target=final_target_proto_active,
                    )
                ),
                use_reliability_eval=use_reliability_eval,
                spatial_k=int(cfg["spatial_k"]),
                sigma_spatial=float(cfg["sigma_spatial"]),
                sigma_spectral=float(cfg["sigma_spectral"]),
                reliability_map_sigma=float(cfg.get("reliability_map_sigma", 1.0)),
            )
        acc_all[i_dataset] = last_result["oa"]
        class_acc_all[i_dataset, :] = last_result["class_accuracy"]
        kappa_all[i_dataset] = last_result["kappa"]

    print("train time per DataSet(s): " + "{:.5f}".format(train_time / max(1, n_datasets)))
    print("test time per DataSet(s): " + "{:.5f}".format(test_time / max(1, n_datasets)))
    print_average_result(acc_all, class_acc_all, kappa_all)
    for i in range(len(acc_all)):
        print("{}:{}".format(i, acc_all[i]))
    print("best acc all={}".format(best_acc_all))
    if len(best_predict_all) == 0:
        print("best predict all is empty; no evaluation interval was reached.")


if __name__ == "__main__":
    main()
