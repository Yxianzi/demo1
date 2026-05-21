# -*- coding:utf-8 -*-
# Author：Mingshuo Cai
# Create_time：2023-08-01
# Updata_time：2024-03-15
# Usage：Implementation of the MLUDA method on the SH2HZ cross-domain dataset

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import mmd
import numpy as np
from sklearn import metrics
from net2 import DSANSS
import time
import utils
from torch.utils.data import TensorDataset, DataLoader
from contrastive_loss import SupConLoss, SafeWeightedSupConLoss
from config_SH2HZ import *
from sklearn import svm
from UtilsCMS import *
from rp_utils import update_ema_teacher
from mv_refine import multiview_refine_pseudo_labels

USE_MVREFINE_V15 = True
USE_EMA_TEACHER = True
USE_CB_RPLS = True
USE_EW_TMCC = False

MV_WARMUP_EPOCHS = 20

MV_THRESHOLD_START = 0.80
MV_THRESHOLD_END = 0.65

MV_PAIR_DELTA = 0.05
MV_PAIR_WEIGHT = 0.7
MV_LMMD_BLEND_MAX = 0.3

RP_EVAL_INTERVAL = 10

RPLS_WARMUP_EPOCHS = MV_WARMUP_EPOCHS
RPLS_LMMD_BLEND_MAX = 0.1

LAMBDA_TGT_CON = 0
TGT_CON_RAMPUP_EPOCHS = 60
LAMBDA_TMCC = 0.01
TMCC_TEMPERATURE = 2.5
TMCC_WARMUP_EPOCHS = 20

def numpy_to_tensor(data, numpy_dtype, torch_dtype):
    array = np.ascontiguousarray(data, dtype=numpy_dtype)
    return torch.frombuffer(array, dtype=torch_dtype).reshape(array.shape)

def get_ema_decay(epoch):
    if epoch <= MV_WARMUP_EPOCHS:
        return 0.99
    if epoch <= 50:
        return 0.995
    return 0.999

def entropy_weighted_mcc_loss(logits, temperature=2.5, sample_weight=None, eps=1e-6):
    prob = F.softmax(logits / temperature, dim=1)
    entropy = -torch.sum(prob * torch.log(prob.clamp_min(eps)), dim=1)
    entropy_weight = 1.0 + torch.exp(-entropy)
    entropy_weight = entropy_weight / entropy_weight.sum().clamp_min(eps) * prob.size(0)

    if sample_weight is not None:
        entropy_weight = entropy_weight * sample_weight.detach().to(
            device=logits.device,
            dtype=logits.dtype
        ).view(-1)
        if entropy_weight.sum().item() <= eps:
            return logits.new_tensor(0.0)
        entropy_weight = entropy_weight / entropy_weight.sum().clamp_min(eps) * prob.size(0)

    weighted_prob = prob * entropy_weight.view(-1, 1)
    class_confusion = torch.mm(weighted_prob.t(), prob)
    class_confusion = class_confusion / class_confusion.sum(dim=1, keepdim=True).clamp_min(eps)

    num_classes = prob.size(1)
    off_diag = class_confusion.sum() - torch.trace(class_confusion)
    return off_diag / num_classes

def evaluate_target_domain(model, test_loader, source_reference_data):
    model.eval()
    total_rewards = 0
    predict = np.array([], dtype=np.int64)
    labels = np.array([], dtype=np.int64)

    with torch.no_grad():
        for test_datas, test_labels in test_loader:
            batch_size = test_labels.shape[0]
            eval_source_data = source_reference_data[:batch_size]

            (_, _, _, _, _,
             _, _, _, test_outputs, _) = model(
                Variable(eval_source_data).cuda(),
                Variable(test_datas).cuda()
            )

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
        where=class_totals != 0
    )
    oa = 100. * total_rewards / len(test_loader.dataset)
    aa = np.mean(class_accuracy)
    kappa = metrics.cohen_kappa_score(labels, predict, labels=class_ids)

    return {
        'total_rewards': total_rewards,
        'total_count': len(test_loader.dataset),
        'predict': predict,
        'labels': labels,
        'oa': oa,
        'aa': aa,
        'kappa': kappa,
        'class_accuracy': class_accuracy
    }

def print_eval_result(name, result):
    print('{}:'.format(name))
    print('\tOA: {}/{} ({:.2f}%)'.format(
        result['total_rewards'],
        result['total_count'],
        result['oa']
    ))
    print('\tAA: {:.2f}%'.format(100 * result['aa']))
    print('\tKappa: {:.4f}'.format(100 * result['kappa']))
    print('\taccuracy for each class:')
    for class_id in range(CLASS_NUM):
        print('\tClass {}: {:.2f}'.format(
            class_id,
            100 * result['class_accuracy'][class_id]
        ))

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

    print('average {} OA: {:.2f} +- {:.2f}'.format(name, oa_mean, oa_std))
    print('average {} AA: {:.2f} +- {:.2f}'.format(name, 100 * aa_mean, 100 * aa_std))
    print('average {} Kappa: {:.4f} +- {:.4f}'.format(name, 100 * kappa_mean, 100 * kappa_std))
    print('{} accuracy for each class:'.format(name))
    for class_id in range(CLASS_NUM):
        print('Class {}: {:.2f} +- {:.2f}'.format(
            class_id,
            100 * class_mean[class_id],
            100 * class_std[class_id]
        ))

##################################
file_path = './datasets/Shanghai-Hangzhou/DataCube.mat'
data_s, data_t, label_s, label_t = utils.cubeData(file_path)

data_s,data_t = ILDA(data_s,data_t,pca_n,radius)

# Loss Function
crossEntropy = nn.CrossEntropyLoss().cuda()
ContrastiveLoss_s = SupConLoss(temperature=0.1).cuda()
ContrastiveLoss_t = SafeWeightedSupConLoss(temperature=0.1).cuda()
DSH_loss = utils.Domain_Occ_loss().cuda()

student_acc = np.zeros([nDataSet, 1])
student_A = np.zeros([nDataSet, CLASS_NUM])
student_k = np.zeros([nDataSet, 1])
teacher_acc = np.zeros([nDataSet, 1])
teacher_A = np.zeros([nDataSet, CLASS_NUM])
teacher_k = np.zeros([nDataSet, 1])
best_predict_all = []
best_acc_all = 0.0
best_G,best_RandPerm,best_Row, best_Column,best_nTrain = None,None,None,None,None

for iDataSet in range(nDataSet):
    print('#######################idataset######################## ', iDataSet)
    utils.set_seed(seeds[iDataSet])

    trainX, trainY = utils.get_sample_data(data_s, label_s, HalfWidth, 180)
    testID, testX, testY, G, RandPerm, Row, Column = utils.get_all_data(data_t, label_t, HalfWidth)

    train_dataset = TensorDataset(
        numpy_to_tensor(trainX, np.float32, torch.float32),
        numpy_to_tensor(trainY, np.int64, torch.long)
    )
    test_dataset = TensorDataset(
        numpy_to_tensor(testX, np.float32, torch.float32),
        numpy_to_tensor(testY, np.int64, torch.long)
    )

    train_loader_s = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    train_loader_t = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=True,drop_last=True)
    test_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False,drop_last=False)

    len_source_loader = len(train_loader_s)
    len_target_loader = len(train_loader_t)

    # model
    feature_encoder = DSANSS(nBand, patch_size, CLASS_NUM).cuda()
    if USE_EMA_TEACHER:
        teacher_encoder = DSANSS(nBand, patch_size, CLASS_NUM).cuda()
        teacher_encoder.load_state_dict(copy.deepcopy(feature_encoder.state_dict()))
        teacher_encoder.eval()
        for param in teacher_encoder.parameters():
            param.requires_grad = False

    print("Training...")

    last_student_accuracy = 0.0
    last_teacher_accuracy = 0.0
    best_student_episdoe = 0
    best_teacher_episdoe = 0
    train_loss = []
    test_acc = []
    running_D_loss, running_F_loss = 0.0, 0.0
    running_label_loss = 0
    running_domain_loss = 0
    total_hit, total_num = 0.0, 0.0
    size = 0.0
    test_acc_list = []

    train_start = time.time()

    #loss plot
    loss1 = []
    loss2 = []
    loss3 = []

    for epoch in range(1, epochs + 1):
        LEARNING_RATE = lr / math.pow((1 + 10 * (epoch - 1) / epochs), 0.75)
        print('learning rate{: .4f}'.format(LEARNING_RATE))
        optimizer = torch.optim.SGD([
            {'params': feature_encoder.feature_layers.parameters(),},
            {'params': feature_encoder.fc1.parameters(), 'lr': LEARNING_RATE},
            {'params': feature_encoder.fc2.parameters(), 'lr': LEARNING_RATE},
            {'params': feature_encoder.head1.parameters(), 'lr': LEARNING_RATE},
            {'params': feature_encoder.head2.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE , momentum=momentum, weight_decay=l2_decay)

        feature_encoder.train()
        if USE_EMA_TEACHER:
            teacher_encoder.eval()

        iter_source = iter(train_loader_s)
        iter_target = iter(train_loader_t)
        num_iter = len_source_loader
        ema_decay = get_ema_decay(epoch) if USE_EMA_TEACHER else 0.0
        target_con_weight = 0.0

        for i in range(1,num_iter):
            source_data, source_label = next(iter_source)
            target_data, _target_label = next(iter_target)

            if i % len_target_loader == 0:
                iter_target = iter(train_loader_t)

            # 0
            source_data0 = utils.radiation_noise(source_data)
            source_data0 = source_data0.type(torch.FloatTensor)
            # 1
            source_data1 = utils.flip_augmentation(source_data)
            # 2
            target_data0 = utils.radiation_noise(target_data)
            target_data0 = target_data0.type(torch.FloatTensor)
            # 3
            target_data1 = utils.flip_augmentation(target_data)

            source_label_cuda = source_label.cuda()
            source_data_cuda = source_data.cuda()
            target_data_cuda = target_data.cuda()

            (source_features, source1, _, source_outputs, source_out,
             target_features,_, target1, target_outputs, target_out) = feature_encoder(source_data_cuda,target_data_cuda)
            (_, source2, _, source_outputs2 ,_,
             _, _, target2, t1, _) = feature_encoder(source_data0.cuda(),target_data0.cuda())
            (_, source3, _, source_outputs3,_,
            _, _, target3, t2, _) =  feature_encoder(source_data1.cuda(),target_data1.cuda())

            student_prob_t = F.softmax(target_outputs.detach(), dim=1)
            _, pseudo_label_t_student = torch.max(student_prob_t, dim=1)

            refined_label = pseudo_label_t_student
            refined_prob = student_prob_t
            reliable_mask = torch.zeros_like(pseudo_label_t_student, dtype=torch.bool)
            sample_weight = target_outputs.new_zeros(target_outputs.size(0))
            pseudo_label_t_for_scl = pseudo_label_t_student
            lmmd_rho = 0.0
            student_hist = torch.bincount(
                pseudo_label_t_student.detach().cpu(), minlength=CLASS_NUM
            )
            mv_stats = {
                "threshold": 0.0,
                "pair_threshold": 0.0,
                "triple_num": 0,
                "pair_num": 0,
                "reliable_num": 0,
                "reliable_ratio": 0.0,
                "teacher_hist": torch.zeros(CLASS_NUM, dtype=torch.long),
                "refined_hist": student_hist,
                "student_hist": student_hist,
                "conf_mean": 0.0,
                "conf_max": 0.0,
                "guard_reject": False,
                "student_nonzero": int((student_hist > 0).sum().item()),
                "refined_nonzero": int((student_hist > 0).sum().item()),
                "student_top_ratio": float((student_hist.max().float() / student_hist.sum().float().clamp_min(1.0)).item()),
                "refined_top_ratio": float((student_hist.max().float() / student_hist.sum().float().clamp_min(1.0)).item()),
                "accepted_num": 0,
                "class_cap": 0,
                "accepted_hist": torch.zeros(CLASS_NUM, dtype=torch.long),
                "accepted_nonzero": 0,
                "accepted_top_ratio": 0.0,
                "accepted_weight_mean": 0.0,
                "accepted_weight_max": 0.0,
                "candidate_num": 0,
                "candidate_ratio": 0.0,
                "per_class_k": 0,
                "topk_ratio": 0.0,
                "teacher_student_agree_num": 0,
                "teacher_stronger_num": 0,
                "rejected_disagree_num": 0,
            }

            if USE_MVREFINE_V15 and USE_EMA_TEACHER and epoch > MV_WARMUP_EPOCHS:
                with torch.no_grad():
                    t_logits0 = teacher_encoder(source_data_cuda, target_data_cuda)[8]
                    t_logits1 = teacher_encoder(source_data_cuda, target_data0.cuda())[8]
                    t_logits2 = teacher_encoder(source_data_cuda, target_data1.cuda())[8]

                    refined_label, refined_prob, reliable_mask, sample_weight, mv_stats = multiview_refine_pseudo_labels(
                        t_logits0,
                        t_logits1,
                        t_logits2,
                        student_prob_t,
                        epoch,
                        epochs,
                        threshold_start=MV_THRESHOLD_START,
                        threshold_end=MV_THRESHOLD_END,
                        pair_delta=MV_PAIR_DELTA,
                        pair_weight=MV_PAIR_WEIGHT,
                    )

                    pseudo_label_t_for_scl = refined_label

            # Supervised Contrastive Loss
            all_source_con_features = torch.cat([source2.unsqueeze(1), source3.unsqueeze(1)],dim=1)
            all_target_con_features = torch.cat([target2.unsqueeze(1), target3.unsqueeze(1)], dim=1)

            # Loss Cls
            cls_loss = crossEntropy(source_outputs, source_label_cuda)
            # Loss Lmmd
            lmmd_loss_base = mmd.lmmd(
                source_features,
                target_features,
                source_label,
                student_prob_t.detach(),
                BATCH_SIZE=BATCH_SIZE,
                CLASS_NUM=CLASS_NUM
            )
            if (
                USE_CB_RPLS
                and epoch > RPLS_WARMUP_EPOCHS
                and reliable_mask.sum().item() > 0
                and sample_weight.detach().sum().item() > 0
            ):
                lmmd_loss_reliable = mmd.weighted_lmmd(
                    source_features,
                    target_features,
                    source_label,
                    refined_prob.detach(),
                    t_weight=sample_weight.detach(),
                    BATCH_SIZE=BATCH_SIZE,
                    CLASS_NUM=CLASS_NUM
                )
                if not torch.isfinite(lmmd_loss_reliable).item():
                    lmmd_loss_reliable = target_features.new_tensor(0.0)

                rpls_progress = min(
                    1.0,
                    max(0.0, (epoch - RPLS_WARMUP_EPOCHS) / max(1, epochs - RPLS_WARMUP_EPOCHS))
                )
                lmmd_rho = RPLS_LMMD_BLEND_MAX * rpls_progress
                lmmd_loss = (1.0 - lmmd_rho) * lmmd_loss_base + lmmd_rho * lmmd_loss_reliable
            else:
                lmmd_loss = lmmd_loss_base
            lambd = 2 / (1 + math.exp(-10 * (epoch) / epochs)) - 1
            if USE_CB_RPLS:
                if epoch > RPLS_WARMUP_EPOCHS:
                    target_con_progress = min(
                        1.0,
                        max(0.0, (epoch - RPLS_WARMUP_EPOCHS) / max(1, TGT_CON_RAMPUP_EPOCHS))
                    )
                    target_con_weight = LAMBDA_TGT_CON * target_con_progress
                else:
                    target_con_weight = 0.0
            else:
                target_con_weight = LAMBDA_TGT_CON
            # Loss Con_s
            contrastive_loss_s = ContrastiveLoss_s(all_source_con_features, source_label)
            # Loss Con_t
            if USE_CB_RPLS:
                target_con_num = int(reliable_mask.sum().item()) if epoch > RPLS_WARMUP_EPOCHS else 0
                if epoch > RPLS_WARMUP_EPOCHS and reliable_mask.sum().item() >= 2:
                    contrastive_loss_t = ContrastiveLoss_t(
                        all_target_con_features[reliable_mask],
                        pseudo_label_t_for_scl[reliable_mask],
                        sample_weight=sample_weight[reliable_mask]
                    )
                else:
                    contrastive_loss_t = target_features.new_tensor(0.0)
            else:
                target_con_num = int(pseudo_label_t_for_scl.numel())
                contrastive_loss_t = ContrastiveLoss_t(
                    all_target_con_features,
                    pseudo_label_t_for_scl
                )
            # Loss Occ
            domain_similar_loss = DSH_loss(source_out, target_out)

            if USE_EW_TMCC and epoch > TMCC_WARMUP_EPOCHS:
                if USE_CB_RPLS and reliable_mask.sum().item() > 0:
                    tmcc_loss = entropy_weighted_mcc_loss(
                        target_outputs,
                        temperature=TMCC_TEMPERATURE,
                        sample_weight=sample_weight
                    )
                else:
                    tmcc_loss = entropy_weighted_mcc_loss(
                        target_outputs,
                        temperature=TMCC_TEMPERATURE,
                        sample_weight=None
                    )
            else:
                tmcc_loss = target_features.new_tensor(0.0)
            if not torch.isfinite(tmcc_loss).item():
                tmcc_loss = target_features.new_tensor(0.0)

            if reliable_mask.sum().item() > 0:
                weight_mean = float(sample_weight[reliable_mask].mean().item())
            else:
                weight_mean = 0.0

            loss_base = (
                cls_loss
                + 0.01 * lambd * lmmd_loss
                + contrastive_loss_s
                + target_con_weight * contrastive_loss_t
                + domain_similar_loss
                + LAMBDA_TMCC * lambd * tmcc_loss
            )

            loss = loss_base

            # Update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if USE_EMA_TEACHER:
                ema_decay = get_ema_decay(epoch)
                update_ema_teacher(feature_encoder, teacher_encoder, decay=ema_decay)
            else:
                ema_decay = 0.0

            pred = source_outputs.data.max(1)[1]
            total_hit += pred.eq(source_label.data.cuda()).sum()
            size += source_label.data.size()[0]

            test_accuracy = 100. * float(total_hit) / size

        print('epoch {:>3d}:   cls loss: {:6.4f}, lmmd loss:{:6f}, con_s loss:{:6f}, con_t loss:{:6f}, domain loss:{:6f}, tmcc loss:{:6f}, lmmd rho:{:6.4f}, target con weight:{:6.4f}, target con num:{:>2d}, reliable weight mean:{:6.4f}, ema decay:{:6.4f}, threshold:{:6.4f}, pair threshold:{:6.4f}, triple num:{:>2d}, pair num:{:>2d}, reliable num:{:>2d}, reliable ratio:{:6.4f}, guard reject:{}, accepted_num:{:>2d}, class cap:{:>2d}, accepted_nonzero:{:>2d}, accepted_top_ratio:{:6.4f}, accepted weight mean:{:6.4f}, candidate num:{:>2d}, candidate ratio:{:6.4f}, per class k:{:>2d}, topk ratio:{:6.4f}, ts agree:{:>2d}, teacher stronger:{:>2d}, rejected disagree:{:>2d}, student_nonzero:{:>2d}, refined_nonzero:{:>2d}, student_top_ratio:{:6.4f}, refined_top_ratio:{:6.4f}, acc {:6.4f}, total loss: {:6.4f}, student_hist:{}, refined_hist:{}, accepted_hist:{}'
              .format(epoch, cls_loss.item(), lmmd_loss.item(), contrastive_loss_s.item(),
               contrastive_loss_t.item(), domain_similar_loss.item(), tmcc_loss.item(), lmmd_rho,
               target_con_weight, target_con_num, weight_mean, ema_decay,
               mv_stats['threshold'], mv_stats['pair_threshold'],
               mv_stats['triple_num'], mv_stats['pair_num'],
               mv_stats['reliable_num'], mv_stats['reliable_ratio'], mv_stats['guard_reject'],
               mv_stats['accepted_num'], mv_stats['class_cap'], mv_stats['accepted_nonzero'],
               mv_stats['accepted_top_ratio'], mv_stats['accepted_weight_mean'],
               mv_stats['candidate_num'], mv_stats['candidate_ratio'], mv_stats['per_class_k'],
               mv_stats['topk_ratio'], mv_stats['teacher_student_agree_num'],
               mv_stats['teacher_stronger_num'], mv_stats['rejected_disagree_num'],
               mv_stats['student_nonzero'],
               mv_stats['refined_nonzero'], mv_stats['student_top_ratio'], mv_stats['refined_top_ratio'],
               total_hit / size,
               loss.item(), mv_stats['student_hist'].tolist(), mv_stats['refined_hist'].tolist(),
               mv_stats['accepted_hist'].tolist()))

        train_end = time.time()
        if epoch % RP_EVAL_INTERVAL == 0 or epoch == epochs:
            print('Testing epoch {} ...'.format(epoch))
            student_result = evaluate_target_domain(feature_encoder, test_loader, source_data)
            student_acc[iDataSet] = student_result['oa']
            student_A[iDataSet, :] = student_result['class_accuracy']
            student_k[iDataSet] = student_result['kappa']
            print_eval_result('Student', student_result)

            if USE_EMA_TEACHER:
                teacher_result = evaluate_target_domain(teacher_encoder, test_loader, source_data)
                teacher_acc[iDataSet] = teacher_result['oa']
                teacher_A[iDataSet, :] = teacher_result['class_accuracy']
                teacher_k[iDataSet] = teacher_result['kappa']
                print_eval_result('EMA Teacher', teacher_result)

            test_end = time.time()

            # Training mode

            if student_result['oa'] > last_student_accuracy:
                # save networks
                # torch.save(feature_encoder.state_dict(),str("../checkpoints/DFSL_feature_encoder_" + "sh2hz_v15_mvrefine" +str(iDataSet) +".pkl"))
                print("save student networks for epoch:", epoch + 1)
                last_student_accuracy = student_result['oa']
                best_student_episdoe = epoch
                best_predict_all = student_result['predict']
                best_G, best_RandPerm, best_Row, best_Column = G, RandPerm, Row, Column
                print('best student epoch:[{}], best student accuracy={}'.format(
                    best_student_episdoe + 1,
                    last_student_accuracy
                ))

            if USE_EMA_TEACHER and teacher_result['oa'] > last_teacher_accuracy:
                print("save teacher networks for epoch:", epoch + 1)
                last_teacher_accuracy = teacher_result['oa']
                best_teacher_episdoe = epoch
                print('best teacher epoch:[{}], best teacher accuracy={}'.format(
                    best_teacher_episdoe + 1,
                    last_teacher_accuracy
                ))

            print('iter:{} best student epoch:[{}], best student accuracy={}'.format(
                iDataSet,
                best_student_episdoe + 1,
                last_student_accuracy
            ))
            if USE_EMA_TEACHER:
                print('iter:{} best teacher epoch:[{}], best teacher accuracy={}'.format(
                    iDataSet,
                    best_teacher_episdoe + 1,
                    last_teacher_accuracy
                ))
            print('***********************************************************************************')

print ("train time per DataSet(s): " + "{:.5f}".format(train_end-train_start))
print("test time per DataSet(s): " + "{:.5f}".format(test_end-train_end))
print_average_result('Student', student_acc, student_A, student_k)
if USE_EMA_TEACHER:
    print_average_result('Teacher', teacher_acc, teacher_A, teacher_k)

best_iDataset = 0
for i in range(len(student_acc)):
    print('{}:{}'.format(i, student_acc[i]))
    if student_acc[i] > student_acc[best_iDataset]:
        best_iDataset = i
print('best student acc all={}'.format(student_acc[best_iDataset]))

if USE_EMA_TEACHER:
    best_teacher_iDataset = 0
    for i in range(len(teacher_acc)):
        print('teacher {}:{}'.format(i, teacher_acc[i]))
        if teacher_acc[i] > teacher_acc[best_teacher_iDataset]:
            best_teacher_iDataset = i
    print('best teacher acc all={}'.format(teacher_acc[best_teacher_iDataset]))

#################classification map################################

for i in range(len(best_predict_all)):  # predict ndarray <class 'tuple'>: (9729,)
    best_G[best_Row[best_RandPerm[ i]]][best_Column[best_RandPerm[ i]]] = best_predict_all[i] + 1

hsi_pic = np.zeros((best_G.shape[0], best_G.shape[1], 3))
for i in range(best_G.shape[0]):
    for j in range(best_G.shape[1]):
        if best_G[i][j] == 0:
            hsi_pic[i, j, :] = [0, 0, 0]
        if best_G[i][j] == 1:
            hsi_pic[i, j, :] = [0, 0, 1]
        if best_G[i][j] == 2:
            hsi_pic[i, j, :] = [0, 1, 0]
        if best_G[i][j] == 3:
            hsi_pic[i, j, :] = [0, 1, 1]
        if best_G[i][j] == 4:
            hsi_pic[i, j, :] = [1, 0, 0]
        if best_G[i][j] == 5:
            hsi_pic[i, j, :] = [1, 0, 1]
        if best_G[i][j] == 6:
            hsi_pic[i, j, :] = [1, 1, 0]
        if best_G[i][j] == 7:
            hsi_pic[i, j, :] = [0.5, 0.5, 1]

# utils.classification_map(hsi_pic[4:-4, 4:-4, :], best_G[4:-4, 4:-4], 24,  "classificationMap/housotn18.png")
