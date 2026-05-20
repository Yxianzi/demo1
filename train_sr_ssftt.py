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
from losses.prototype_alignment import (
    build_fused_prototypes,
    spatial_reliable_prototype_alignment,
)
from losses.spatial_reliability import (
    boundary_preserving_spatial_consistency,
    build_target_coordinates,
    compute_spatial_reliability,
    spatial_sort_indices,
)
from models.sr_ssftt import SRSSFTT
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


def spatial_voting_refinement(base_probs, affinity, alpha):
    """Blend sample probabilities with spatial-spectral neighbor votes."""

    row_sum = affinity.sum(dim=1, keepdim=True)
    neighbor_probs = torch.matmul(affinity, base_probs) / row_sum.clamp_min(1e-8)
    neighbor_probs = torch.where(row_sum > 0, neighbor_probs, base_probs)
    refined = (1.0 - alpha) * base_probs + alpha * neighbor_probs
    return refined / refined.sum(dim=1, keepdim=True).clamp_min(1e-8)


def class_balanced_reliable_selection(labels, scores, threshold, topk_per_class, num_classes):
    """Select at most top-k reliable target samples for each pseudo class."""

    selected = torch.zeros_like(labels, dtype=torch.bool)
    candidate = scores > threshold
    for class_id in range(num_classes):
        class_mask = candidate & (labels == class_id)
        index = torch.nonzero(class_mask, as_tuple=False).view(-1)
        if index.numel() == 0:
            continue
        class_scores = scores[index]
        keep = min(int(topk_per_class), index.numel())
        order = torch.argsort(class_scores, descending=True)[:keep]
        selected[index[order]] = True
    return selected


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


def evaluate_target_domain(model, test_loader, device, num_classes, prototypes=None):
    model.eval()
    predict = np.array([], dtype=np.int64)
    labels = np.array([], dtype=np.int64)
    total_hit = 0

    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                data, label, _coords = batch
            else:
                data, label = batch
            data = data.to(device)
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

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(cfg["lr"]),
            momentum=float(cfg["momentum"]),
            weight_decay=float(cfg["weight_decay"]),
        )

        current_prototypes = None
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
                "reliability_mean": 0.0,
                "reliability_max": 0.0,
                "selected_num": 0,
                "loss": 0.0,
            }
            selected_per_class = torch.zeros(num_classes, dtype=torch.long)
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
                prototypes_for_head = None if args.disable_proto_head else current_prototypes

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
                    target_reliability_for_tokenizer = None
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

                target_out = model(
                    target_x,
                    reliability=target_reliability_for_tokenizer,
                    domain="target",
                    prototypes=prototypes_for_head,
                )
                source_features = source_out["features"]
                target_features = target_out["features"]
                source_logits = source_out["logits"]
                target_logits = target_out["logits"]
                target_probs = F.softmax(target_logits, dim=1)

                with torch.no_grad():
                    refinement_base = 0.5 * target_probs.detach() + 0.5 * teacher_probs.detach()
                    refined_probs = spatial_voting_refinement(
                        refinement_base,
                        affinity.detach(),
                        alpha=float(cfg["spatial_vote_alpha"]),
                    )
                    pseudo_labels = refined_probs.argmax(dim=1)
                    selected_mask = class_balanced_reliable_selection(
                        pseudo_labels,
                        reliability.detach(),
                        threshold,
                        int(cfg["topk_per_class"]),
                        num_classes,
                    )

                source_ce = cross_entropy(source_logits, source_y_cuda)
                lmmd_loss = mmd.lmmd(
                    source_features,
                    target_features,
                    source_y,
                    refined_probs.detach(),
                    BATCH_SIZE=batch_size,
                    CLASS_NUM=num_classes,
                )
                mcc_loss = minimum_class_confusion_loss(target_logits)

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

                teacher_consistency_loss = weighted_kl_loss(
                    target_logits,
                    refined_probs.detach(),
                    weight=reliability.detach(),
                    mask=selected_mask if selected_mask.any() else None,
                )

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
                total_loss = loss_warm + ramp * loss_extra

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                update_ema_teacher(model, teacher_model, decay=float(cfg["ema_decay"]))

                with torch.no_grad():
                    current_prototypes = build_fused_prototypes(
                        source_features.detach(),
                        source_y_cuda.detach(),
                        target_features.detach(),
                        pseudo_labels.detach(),
                        reliability.detach(),
                        num_classes,
                        target_mask=selected_mask.detach(),
                        eta=float(cfg.get("prototype_eta", 0.7)),
                    ).detach()

                selected_hist = torch.bincount(
                    pseudo_labels[selected_mask].detach().cpu(),
                    minlength=num_classes,
                )
                selected_per_class += selected_hist
                meter["source_ce"] += float(source_ce.item())
                meter["lmmd"] += float(lmmd_loss.item())
                meter["mcc"] += float(mcc_loss.item())
                meter["pseudo_ce"] += float(pseudo_ce_loss.item())
                meter["proto"] += float(proto_align_loss.item())
                meter["sp_pred"] += float(spatial_pred_loss.item())
                meter["sp_feat"] += float(spatial_feat_loss.item())
                meter["cons"] += float(teacher_consistency_loss.item())
                meter["reliability_mean"] += float(spatial_stats["reliability_mean"])
                meter["reliability_max"] += float(spatial_stats["reliability_max"])
                meter["selected_num"] += int(selected_mask.sum().item())
                meter["loss"] += float(total_loss.item())
                num_batches += 1

            denom = max(1, num_batches)
            print(
                "epoch {:>3d}: lr:{:.5f}, source_ce:{:6.4f}, lmmd_loss:{:6.4f}, "
                "mcc_loss:{:6.4f}, pseudo_ce_loss:{:6.4f}, proto_align_loss:{:6.4f}, "
                "spatial_pred_loss:{:6.4f}, spatial_feat_loss:{:6.4f}, "
                "teacher_consistency_loss:{:6.4f}, reliability_mean:{:6.4f}, "
                "reliability_max:{:6.4f}, selected_pseudo_num:{:>4d}, "
                "selected_per_class:{}, threshold:{:6.4f}, ramp:{:6.4f}, total_loss:{:6.4f}".format(
                    epoch,
                    learning_rate,
                    meter["source_ce"] / denom,
                    meter["lmmd"] / denom,
                    meter["mcc"] / denom,
                    meter["pseudo_ce"] / denom,
                    meter["proto"] / denom,
                    meter["sp_pred"] / denom,
                    meter["sp_feat"] / denom,
                    meter["cons"] / denom,
                    meter["reliability_mean"] / denom,
                    meter["reliability_max"] / denom,
                    meter["selected_num"],
                    selected_per_class.tolist(),
                    threshold,
                    ramp,
                    meter["loss"] / denom,
                )
            )

            if epoch % eval_interval == 0 or epoch == epochs:
                test_begin = time.time()
                print("Testing epoch {} ...".format(epoch))
                eval_prototypes = None if args.disable_proto_head else current_prototypes
                last_result = evaluate_target_domain(
                    model,
                    target_test_loader,
                    device,
                    num_classes,
                    prototypes=eval_prototypes,
                )
                print_eval_result(last_result)
                test_time += time.time() - test_begin
                if last_result["oa"] > best_acc_all:
                    best_acc_all = last_result["oa"]
                    best_predict_all = last_result["predict"]

        train_time += time.time() - train_start
        if last_result is None:
            last_result = evaluate_target_domain(
                model,
                target_test_loader,
                device,
                num_classes,
                prototypes=None if args.disable_proto_head else current_prototypes,
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
