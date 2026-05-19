# -*- coding:utf-8 -*-
# Author：Mingshuo Cai
# Create_time：2023-08-01
# Updata_time：2024-03-15
# Usage：Implementation of the MLUDA method on the Houston cross-domain dataset

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
from contrastive_loss import SupConLoss
from config_Houston import *
from sklearn import svm
from UtilsCMS import *
from rp_utils import update_ema_teacher, get_reliable_pseudo_labels
from prototype_memory import PrototypeMemory, prototype_contrastive_loss

USE_RP = True
RP_FEATURE_DIM = 288
RP_WARMUP_EPOCHS = 30
RP_THRESHOLD_START = 0.50
RP_THRESHOLD_END = 0.80
RP_PROTO_WEIGHT = 0.01
RP_MIN_RELIABLE_SAMPLES = 4
RP_EVAL_INTERVAL = 10

def numpy_to_tensor(data, numpy_dtype, torch_dtype):
    array = np.ascontiguousarray(data, dtype=numpy_dtype)
    return torch.frombuffer(array, dtype=torch_dtype).reshape(array.shape)

##################################
data_path_s = './datasets/Houston/Houston13.mat'
label_path_s = './datasets/Houston/Houston13_7gt.mat'
data_path_t = './datasets/Houston/Houston18.mat'
label_path_t = './datasets/Houston/Houston18_7gt.mat'

data_s,label_s = utils.load_data_houston(data_path_s,label_path_s)
data_t,label_t = utils.load_data_houston(data_path_t,label_path_t)

data_s,data_t = ILDA(data_s,data_t,pca_n,radius)

# Loss Function
crossEntropy = nn.CrossEntropyLoss().cuda()
ContrastiveLoss_s = SupConLoss(temperature=0.1).cuda()
ContrastiveLoss_t = SupConLoss(temperature=0.1).cuda()
DSH_loss = utils.Domain_Occ_loss().cuda()

acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, CLASS_NUM])
k = np.zeros([nDataSet, 1])
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
    if USE_RP:
        teacher_encoder = DSANSS(nBand, patch_size, CLASS_NUM).cuda()
        teacher_encoder.load_state_dict(copy.deepcopy(feature_encoder.state_dict()))
        teacher_encoder.eval()
        for param in teacher_encoder.parameters():
            param.requires_grad = False

        source_memory = PrototypeMemory(CLASS_NUM, RP_FEATURE_DIM, momentum=0.9).cuda()

    print("Training...")

    last_accuracy = 0.0
    best_episdoe = 0
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
        if USE_RP:
            teacher_encoder.eval()

        iter_source = iter(train_loader_s)
        iter_target = iter(train_loader_t)
        num_iter = len_source_loader

        for i in range(1,num_iter):
            source_data, source_label = next(iter_source)
            target_data, target_label = next(iter_target)

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

            softmax_output_t = F.softmax(target_outputs.detach(), dim=1)
            _, pseudo_label_t_student = torch.max(softmax_output_t, dim=1)

            if USE_RP:
                source_memory.update(source_features.detach(), source_label_cuda)
                with torch.no_grad():
                    (_, _, _, _, _,
                     _, _, _, teacher_target_outputs, _) = teacher_encoder(source_data_cuda,target_data_cuda)
                    conf_t, pseudo_label_t_teacher, reliable_mask, prob_t_teacher = get_reliable_pseudo_labels(
                        teacher_target_outputs, epoch, epochs,
                        threshold_start=RP_THRESHOLD_START,
                        threshold_end=RP_THRESHOLD_END
                    )

                proto_loss_t2s = source_features.new_tensor(0.0)
                proto_weight = 0.0
                reliable_num = reliable_mask.sum().item()
                reliable_ratio = reliable_mask.float().mean().item()
                conf_mean = conf_t.mean().item()
                conf_max = conf_t.max().item()
                pseudo_hist = torch.bincount(
                    pseudo_label_t_teacher.detach().cpu(), minlength=CLASS_NUM
                )

                if epoch > RP_WARMUP_EPOCHS and reliable_num >= RP_MIN_RELIABLE_SAMPLES:
                    proto_loss_t2s = prototype_contrastive_loss(
                        target_features,
                        pseudo_label_t_teacher,
                        source_memory.get(),
                        temperature=0.1,
                        mask=reliable_mask
                    )
                    progress = (epoch - RP_WARMUP_EPOCHS) / max(1, epochs - RP_WARMUP_EPOCHS)
                    proto_weight = RP_PROTO_WEIGHT * min(1.0, progress)
            else:
                proto_loss_t2s = source_features.new_tensor(0.0)
                proto_weight = 0.0

            # Supervised Contrastive Loss
            all_source_con_features = torch.cat([source2.unsqueeze(1), source3.unsqueeze(1)],dim=1)
            all_target_con_features = torch.cat([target2.unsqueeze(1), target3.unsqueeze(1)], dim=1)

            # Loss Cls
            cls_loss = crossEntropy(source_outputs, source_label_cuda)
            # Loss Lmmd
            lmmd_loss = mmd.lmmd(source_features, target_features, source_label,
                                 softmax_output_t, BATCH_SIZE=BATCH_SIZE,
                                 CLASS_NUM=CLASS_NUM)
            lambd = 2 / (1 + math.exp(-10 * (epoch) / epochs)) - 1
            # Loss Con_s
            contrastive_loss_s = ContrastiveLoss_s(all_source_con_features, source_label)
            # Loss Con_t
            contrastive_loss_t = ContrastiveLoss_t(all_target_con_features, pseudo_label_t_student)
            # Loss Occ
            domain_similar_loss = DSH_loss(source_out, target_out)

            loss_base = (
                cls_loss
                + 0.01 * lambd * lmmd_loss
                + contrastive_loss_s
                + contrastive_loss_t
                + domain_similar_loss
            )

            if USE_RP:
                loss = loss_base + proto_weight * proto_loss_t2s
            else:
                loss = loss_base

            # Update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if USE_RP:
                update_ema_teacher(feature_encoder, teacher_encoder, decay=0.999)

            pred = source_outputs.data.max(1)[1]
            total_hit += pred.eq(source_label.data.cuda()).sum()
            size += source_label.data.size()[0]

            test_accuracy = 100. * float(total_hit) / size

        if USE_RP:
            print('epoch {:>3d}:   cls loss: {:6.4f},lmmd loss:{:6f},con_s loss:{:6f}, con_t loss:{:6f}, proto_t2s loss:{:6f}, proto weight:{:6.4f}, reliable ratio:{:6.4f}, reliable num:{:>2d}, conf mean:{:6.4f}, conf max:{:6.4f}, acc {:6.4f}, total loss: {:6.4f}, pseudo hist:{}'
                  .format(epoch , cls_loss.item(),lmmd_loss.item(),contrastive_loss_s.item(),contrastive_loss_t.item(),
                   proto_loss_t2s.item(),proto_weight,reliable_ratio,int(reliable_num),conf_mean,conf_max,total_hit / size,loss.item(),pseudo_hist.tolist()))
        else:
            print('epoch {:>3d}:   cls loss: {:6.4f},lmmd loss:{:6f},con_s loss:{:6f}, con_t loss:{:6f},acc {:6.4f}, total loss: {:6.4f}'
                  .format(epoch , cls_loss.item(),lmmd_loss.item(),contrastive_loss_s.item(),contrastive_loss_t.item(),
                   total_hit / size,loss.item()))

        train_end = time.time()
        if epoch % RP_EVAL_INTERVAL == 0 or epoch == epochs:
            # print("Testing ...")
            feature_encoder.eval()
            total_rewards = 0
            counter = 0
            accuracies = []
            predict = np.array([], dtype=np.int64)
            labels = np.array([], dtype=np.int64)
            with torch.no_grad():
                for test_datas, test_labels in test_loader:
                    batch_size = test_labels.shape[0]

                    source_features, source1, _, source_outputs, source_out, test_features, _, _, test_outputs, _ = feature_encoder(
                            Variable(source_data).cuda(), Variable(test_datas).cuda())

                    pred = test_outputs.data.max(1)[1]

                    pred_np = np.asarray(pred.detach().cpu().tolist(), dtype=np.int64)
                    test_labels = np.asarray(test_labels.cpu().tolist(), dtype=np.int64)
                    rewards = [1 if pred_np[j] == test_labels[j] else 0 for j in range(batch_size)]

                    total_rewards += np.sum(rewards)
                    counter += batch_size

                    predict = np.append(predict, pred_np)
                    labels = np.append(labels, test_labels)

                    accuracy = total_rewards / 1.0 / counter  #
                    accuracies.append(accuracy)

            test_accuracy = 100. * total_rewards / len(test_loader.dataset)
            acc[iDataSet] = 100. * total_rewards / len(test_loader.dataset)
            OA = acc
            C = metrics.confusion_matrix(labels, predict)
            A[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=np.float64)

            k[iDataSet] = metrics.cohen_kappa_score(labels, predict)
            print('\t\tAccuracy: {}/{} ({:.2f}%)\n'.format(total_rewards, len(test_loader.dataset),
                                                           100. * total_rewards / len(test_loader.dataset)))
            test_end = time.time()

            # Training mode

            if test_accuracy > last_accuracy:
                # save networks
                # torch.save(feature_encoder.state_dict(),str("../checkpoints/DFSL_feature_encoder_" + "houston_cl_lmmd_dis_attention" +str(iDataSet) +".pkl"))
                print("save networks for epoch:", epoch + 1)
                last_accuracy = test_accuracy
                best_episdoe = epoch
                best_predict_all = predict
                best_G, best_RandPerm, best_Row, best_Column = G, RandPerm, Row, Column
                print('best epoch:[{}], best accuracy={}'.format(best_episdoe + 1, last_accuracy))

            print('iter:{} best epoch:[{}], best accuracy={}'.format(iDataSet, best_episdoe + 1, last_accuracy))
            print('***********************************************************************************')

AA = np.mean(A, 1)
AAMean = np.mean(AA,0)
AAStd = np.std(AA)
AMean = np.mean(A, 0)
AStd = np.std(A, 0)
OAMean = np.mean(acc)
OAStd = np.std(acc)
kMean = np.mean(k)
kStd = np.std(k)
print ("train time per DataSet(s): " + "{:.5f}".format(train_end-train_start))
print("test time per DataSet(s): " + "{:.5f}".format(test_end-train_end))
print ("average OA: " + "{:.2f}".format( OAMean) + " +- " + "{:.2f}".format( OAStd))
print ("average AA: " + "{:.2f}".format(100 * AAMean) + " +- " + "{:.2f}".format(100 * AAStd))
print ("average kappa: " + "{:.4f}".format(100 *kMean) + " +- " + "{:.4f}".format(100 *kStd))
print ("accuracy for each class: ")
for i in range(CLASS_NUM):
    print ("Class " + str(i) + ": " + "{:.2f}".format(100 * AMean[i]) + " +- " + "{:.2f}".format(100 * AStd[i]))

best_iDataset = 0
for i in range(len(acc)):
    print('{}:{}'.format(i, acc[i]))
    if acc[i] > acc[best_iDataset]:
        best_iDataset = i
print('best acc all={}'.format(acc[best_iDataset]))

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
