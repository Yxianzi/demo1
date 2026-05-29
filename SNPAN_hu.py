# -*- coding:utf-8 -*-
"""SNPAN on Houston13 -> Houston18.

Spatial Neighborhood Preserving and Prototype Alignment Network.
This script is intentionally independent from the MLUDA mv-refine/RPLS code.
"""

import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.data import DataLoader, TensorDataset

import utils
from UtilsCMS import ILDA
from config_Houston import *
from snpan_losses import (
    class_prototype_alignment_loss,
    spatial_neighborhood_preserving_loss,
)


class SNPANBackbone(nn.Module):
    """A compact spectral-spatial CNN for HSI patch classification."""

    def __init__(self, n_band, patch_size, num_classes, feature_dim=128):
        super(SNPANBackbone, self).__init__()
        self.n_outputs = feature_dim
        self.spectral = nn.Sequential(
            nn.Conv3d(1, 24, kernel_size=(7, 1, 1), padding=(3, 0, 0), bias=False),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True),
            nn.Conv3d(24, 32, kernel_size=(7, 1, 1), padding=(3, 0, 0), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.spectral_reduce = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(n_band, 1, 1), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, feature_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(feature_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = x.float().unsqueeze(1)
        x = self.spectral(x)
        x = self.spectral_reduce(x).squeeze(2)
        x = self.spatial(x)
        features = self.pool(x).flatten(1)
        logits = self.classifier(features)
        return logits, features


def parse_args():
    parser = argparse.ArgumentParser(description="SNPAN Houston13 -> Houston18")
    parser.add_argument("--lambda_spa", type=float, default=0.05)
    parser.add_argument("--lambda_proto", type=float, default=0.1)
    parser.add_argument("--warmup_epoch", type=int, default=20)
    parser.add_argument("--ramp_epoch", type=int, default=10)
    parser.add_argument("--neighbor_k", type=int, default=4)
    parser.add_argument("--conf_th", type=float, default=0.7)

    parser.add_argument("--epochs", type=int, default=epochs)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--n_dataset", type=int, default=nDataSet)
    parser.add_argument("--num_per_class", type=int, default=180)
    parser.add_argument("--lr", type=float, default=lr)
    parser.add_argument("--cuda_id", type=str, default=cuda_id)
    parser.add_argument("--no_cuda", action="store_true", default=no_cuda)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    return parser.parse_args()


def numpy_to_tensor(data, numpy_dtype, torch_dtype):
    array = np.ascontiguousarray(data, dtype=numpy_dtype)
    return torch.frombuffer(array, dtype=torch_dtype).reshape(array.shape)


def build_target_coordinates(row, column, rand_perm, half_width):
    row = np.asarray(row)[np.asarray(rand_perm)] - int(half_width)
    column = np.asarray(column)[np.asarray(rand_perm)] - int(half_width)
    coords = np.stack([row, column], axis=1)
    return numpy_to_tensor(coords, np.float32, torch.float32)


def evaluate_target_domain(model, test_loader, device):
    model.eval()
    total_rewards = 0
    predict = np.array([], dtype=np.int64)
    labels = np.array([], dtype=np.int64)

    with torch.no_grad():
        for test_datas, test_labels in test_loader:
            test_datas = test_datas.to(device)
            test_outputs, _features = model(test_datas)
            pred = test_outputs.data.max(1)[1]

            pred_np = np.asarray(pred.detach().cpu().tolist(), dtype=np.int64)
            labels_np = np.asarray(test_labels.cpu().tolist(), dtype=np.int64)

            total_rewards += np.sum(pred_np == labels_np)
            predict = np.append(predict, pred_np)
            labels = np.append(labels, labels_np)

    class_ids = np.arange(CLASS_NUM)
    confusion = metrics.confusion_matrix(labels, predict, labels=class_ids)
    class_totals = np.sum(confusion, axis=1, dtype=np.float64)
    class_accuracy = np.divide(
        np.diag(confusion),
        class_totals,
        out=np.zeros(CLASS_NUM, dtype=np.float64),
        where=class_totals != 0,
    )
    oa = 100.0 * total_rewards / len(test_loader.dataset)
    aa = np.mean(class_accuracy)
    kappa = metrics.cohen_kappa_score(labels, predict, labels=class_ids)

    return {
        "total_rewards": total_rewards,
        "total_count": len(test_loader.dataset),
        "predict": predict,
        "labels": labels,
        "oa": oa,
        "aa": aa,
        "kappa": kappa,
        "class_accuracy": class_accuracy,
    }


def print_eval_result(name, result):
    print("{}:".format(name))
    print(
        "\tOA: {}/{} ({:.2f}%)".format(
            result["total_rewards"],
            result["total_count"],
            result["oa"],
        )
    )
    print("\tAA: {:.2f}%".format(100 * result["aa"]))
    print("\tKappa: {:.4f}".format(100 * result["kappa"]))
    print("\taccuracy for each class:")
    for class_id in range(CLASS_NUM):
        print(
            "\tClass {}: {:.2f}".format(
                class_id,
                100 * result["class_accuracy"][class_id],
            )
        )


def print_average_result(name, acc_values, class_acc_values, kappa_values):
    aa = np.mean(class_acc_values, 1)
    aa_mean = np.mean(aa, 0)
    aa_std = np.std(aa)
    class_mean = np.mean(class_acc_values, 0)
    class_std = np.std(class_acc_values, 0)
    oa_mean = np.mean(acc_values)
    oa_std = np.std(acc_values)
    kappa_mean = np.mean(kappa_values)
    kappa_std = np.std(kappa_values)

    print("average {} OA: {:.2f} +- {:.2f}".format(name, oa_mean, oa_std))
    print("average {} AA: {:.2f} +- {:.2f}".format(name, 100 * aa_mean, 100 * aa_std))
    print("average {} Kappa: {:.4f} +- {:.4f}".format(name, 100 * kappa_mean, 100 * kappa_std))
    print("{} accuracy for each class:".format(name))
    for class_id in range(CLASS_NUM):
        print(
            "Class {}: {:.2f} +- {:.2f}".format(
                class_id,
                100 * class_mean[class_id],
                100 * class_std[class_id],
            )
        )


def train_one_dataset(args, i_dataset, data_s, label_s, data_t, label_t, device):
    print("#######################idataset######################## ", i_dataset)
    print(
        "SNPAN controls: lambda_spa={}, lambda_proto={}, warmup_epoch={}, "
        "ramp_epoch={}, neighbor_k={}, conf_th={}".format(
            args.lambda_spa,
            args.lambda_proto,
            args.warmup_epoch,
            args.ramp_epoch,
            args.neighbor_k,
            args.conf_th,
        )
    )
    utils.set_seed(seeds[i_dataset])

    train_x, train_y = utils.get_sample_data(data_s, label_s, HalfWidth, args.num_per_class)
    test_id, test_x, test_y, g_map, rand_perm, row, column = utils.get_all_data(
        data_t,
        label_t,
        HalfWidth,
    )
    _ = test_id
    target_coords = build_target_coordinates(row, column, rand_perm, HalfWidth)

    train_dataset = TensorDataset(
        numpy_to_tensor(train_x, np.float32, torch.float32),
        numpy_to_tensor(train_y, np.int64, torch.long),
    )
    target_dataset = TensorDataset(
        numpy_to_tensor(test_x, np.float32, torch.float32),
        numpy_to_tensor(test_y, np.int64, torch.long),
        target_coords,
    )
    test_dataset = TensorDataset(
        numpy_to_tensor(test_x, np.float32, torch.float32),
        numpy_to_tensor(test_y, np.int64, torch.long),
    )

    train_loader_s = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    train_loader_t = DataLoader(
        target_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    model = SNPANBackbone(nBand, patch_size, CLASS_NUM).to(device)
    cross_entropy = nn.CrossEntropyLoss().to(device)

    print("Training...")
    best_accuracy = 0.0
    best_epoch = 0
    best_predict = None
    best_path = os.path.join(args.save_dir, "SNPAN_hu_dataset{}_best.pth".format(i_dataset))
    train_start = time.time()
    test_end = train_start

    for epoch in range(1, args.epochs + 1):
        learning_rate = args.lr / math.pow((1 + 10 * (epoch - 1) / args.epochs), 0.75)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=l2_decay,
        )
        print("learning rate{: .4f}".format(learning_rate))

        model.train()
        iter_target = iter(train_loader_t)
        batch_count = 0
        total_hit = 0
        total_num = 0
        loss_sum = 0.0
        cls_sum = 0.0
        spa_sum = 0.0
        proto_sum = 0.0
        last_spa_stats = {"spa_edges": 0, "spa_affinity": 0.0, "spa_agree": 0.0, "spa_conf": 0.0}
        last_proto_stats = {
            "proto_confident": 0,
            "proto_ratio": 0.0,
            "proto_classes": 0,
            "proto_hist": [0] * CLASS_NUM,
        }

        for source_data, source_label in train_loader_s:
            try:
                target_data, _target_label, target_coord = next(iter_target)
            except StopIteration:
                iter_target = iter(train_loader_t)
                target_data, _target_label, target_coord = next(iter_target)

            source_data = source_data.to(device)
            source_label = source_label.to(device)
            target_data = target_data.to(device)
            target_coord = target_coord.to(device)

            source_logits, source_features = model(source_data)
            target_logits, target_features = model(target_data)

            cls_loss = cross_entropy(source_logits, source_label)
            if epoch > args.warmup_epoch:
                spa_loss, last_spa_stats = spatial_neighborhood_preserving_loss(
                    target_features,
                    target_logits,
                    target_coord,
                    target_patches=target_data,
                    neighbor_k=args.neighbor_k,
                )
                proto_loss, last_proto_stats = class_prototype_alignment_loss(
                    source_features,
                    source_label,
                    target_features,
                    target_logits,
                    CLASS_NUM,
                    conf_th=args.conf_th,
                )
            else:
                spa_loss = source_features.sum() * 0.0
                proto_loss = source_features.sum() * 0.0

            if epoch > args.warmup_epoch:
                adapt_progress = min(
                    1.0,
                    max(
                        0.0,
                        (epoch - args.warmup_epoch) / max(1, int(args.ramp_epoch)),
                    ),
                )
            else:
                adapt_progress = 0.0
            spa_weight = args.lambda_spa * adapt_progress
            proto_weight = args.lambda_proto * adapt_progress

            loss = cls_loss + spa_weight * spa_loss + proto_weight * proto_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = source_logits.data.max(1)[1]
            total_hit += int(pred.eq(source_label).sum().item())
            total_num += int(source_label.size(0))
            batch_count += 1
            loss_sum += float(loss.item())
            cls_sum += float(cls_loss.item())
            spa_sum += float(spa_loss.item())
            proto_sum += float(proto_loss.item())

        train_end = time.time()
        source_acc = total_hit / max(total_num, 1)
        avg_loss = loss_sum / max(batch_count, 1)
        avg_cls = cls_sum / max(batch_count, 1)
        avg_spa = spa_sum / max(batch_count, 1)
        avg_proto = proto_sum / max(batch_count, 1)

        result = evaluate_target_domain(model, test_loader, device)
        test_end = time.time()

        print(
            "epoch {:>3d}: cls loss: {:6.4f}, spa loss:{:6.4f}, "
            "spa weight:{:6.4f}, proto loss:{:6.4f}, proto weight:{:6.4f}, "
            "acc {:6.4f}, total loss: {:6.4f}, "
            "spa edges:{:>3d}, spa affinity:{:6.4f}, spa agree:{:6.4f}, "
            "target conf:{:6.4f}, proto confident:{:>3d}, proto ratio:{:6.4f}, "
            "proto classes:{:>2d}, pseudo hist:{}".format(
                epoch,
                avg_cls,
                avg_spa,
                spa_weight,
                avg_proto,
                proto_weight,
                source_acc,
                avg_loss,
                last_spa_stats.get("spa_edges", 0),
                last_spa_stats.get("spa_affinity", 0.0),
                last_spa_stats.get("spa_agree", 0.0),
                last_spa_stats.get("spa_conf", 0.0),
                last_proto_stats.get("proto_confident", 0),
                last_proto_stats.get("proto_ratio", 0.0),
                last_proto_stats.get("proto_classes", 0),
                last_proto_stats.get("proto_hist", [0] * CLASS_NUM),
            )
        )
        print(
            "\tTarget OA: {}/{} ({:.2f}%), AA: {:.2f}%, Kappa: {:.4f}".format(
                result["total_rewards"],
                result["total_count"],
                result["oa"],
                100 * result["aa"],
                100 * result["kappa"],
            )
        )

        if result["oa"] > best_accuracy:
            best_accuracy = result["oa"]
            best_epoch = epoch
            best_predict = result["predict"]
            torch.save(
                {
                    "epoch": epoch,
                    "oa": result["oa"],
                    "aa": result["aa"],
                    "kappa": result["kappa"],
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                },
                best_path,
            )
            print("save networks for epoch:", epoch)
            print("best epoch:[{}], best accuracy={}".format(best_epoch, best_accuracy))

        print("iter:{} best epoch:[{}], best accuracy={}".format(i_dataset, best_epoch, best_accuracy))
        print("***********************************************************************************")

    print("Best Target result for dataset {}:".format(i_dataset))
    best_result = evaluate_target_domain(model, test_loader, device)
    if best_predict is not None and os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        best_result = evaluate_target_domain(model, test_loader, device)
    print_eval_result("Target", best_result)

    return {
        "oa": best_result["oa"],
        "class_accuracy": best_result["class_accuracy"],
        "kappa": best_result["kappa"],
        "predict": best_predict,
        "g_map": g_map,
        "rand_perm": rand_perm,
        "row": row,
        "column": column,
        "train_time": train_end - train_start,
        "test_time": test_end - train_end,
        "best_path": best_path,
    }


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id
    os.makedirs(args.save_dir, exist_ok=True)

    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda" if use_cuda else "cpu")
    print("device:", device)

    data_path_s = "./datasets/Houston/Houston13.mat"
    label_path_s = "./datasets/Houston/Houston13_7gt.mat"
    data_path_t = "./datasets/Houston/Houston18.mat"
    label_path_t = "./datasets/Houston/Houston18_7gt.mat"

    data_s, label_s = utils.load_data_houston(data_path_s, label_path_s)
    data_t, label_t = utils.load_data_houston(data_path_t, label_path_t)
    data_s, data_t = ILDA(data_s, data_t, pca_n, radius)

    dataset_num = min(args.n_dataset, len(seeds))
    acc = np.zeros([dataset_num, 1])
    class_acc = np.zeros([dataset_num, CLASS_NUM])
    kappa = np.zeros([dataset_num, 1])
    train_times = []
    test_times = []

    best_acc_all = 0.0
    best_path_all = ""
    for i_dataset in range(dataset_num):
        result = train_one_dataset(args, i_dataset, data_s, label_s, data_t, label_t, device)
        acc[i_dataset] = result["oa"]
        class_acc[i_dataset, :] = result["class_accuracy"]
        kappa[i_dataset] = result["kappa"]
        train_times.append(result["train_time"])
        test_times.append(result["test_time"])
        if result["oa"] > best_acc_all:
            best_acc_all = result["oa"]
            best_path_all = result["best_path"]

    print("train time per DataSet(s): " + "{:.5f}".format(np.mean(train_times)))
    print("test time per DataSet(s): " + "{:.5f}".format(np.mean(test_times)))
    print_average_result("SNPAN", acc, class_acc, kappa)

    best_i_dataset = 0
    for i_dataset in range(len(acc)):
        print("{}:{}".format(i_dataset, acc[i_dataset]))
        if acc[i_dataset] > acc[best_i_dataset]:
            best_i_dataset = i_dataset
    print("best acc all={}".format(acc[best_i_dataset]))
    print("best model path={}".format(best_path_all))


if __name__ == "__main__":
    main()
