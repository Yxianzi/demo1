import numpy as np

import torch
import torch.nn.functional as F

# s_vec_label = convert_to_onehot(s_sca_label, CLASS_NUM)


def convert_to_onehot(sca_label, class_num):
    return np.eye(class_num)[sca_label]

class Weight:

    @staticmethod
    # weight_ss, weight_tt, weight_st = Weight.cal_weight(s_label, t_label, type='visual',batch_size=BATCH_SIZE, class_num=CLASS_NUM)

    def cal_weight(s_label, t_label, batch_size,CLASS_NUM):
        batch_size = s_label.size()[0]
        device = t_label.device if torch.is_tensor(t_label) else s_label.device
        s_sca_label_t = s_label.to(device=device, dtype=torch.long).view(-1)
        t_vec_label_t = t_label.detach().to(device=device, dtype=torch.float32)
        t_sca_label_t = t_vec_label_t.max(1)[1]

        s_onehot = F.one_hot(s_sca_label_t, num_classes=CLASS_NUM).float()
        s_sum = s_onehot.sum(dim=0, keepdim=True)
        s_sum = torch.where(s_sum == 0, torch.full_like(s_sum, 100), s_sum)
        s_vec_label_t = s_onehot / s_sum

        t_sum = t_vec_label_t.sum(dim=0, keepdim=True)
        t_sum = torch.where(t_sum == 0, torch.full_like(t_sum, 100), t_sum)
        t_vec_label_t = t_vec_label_t / t_sum

        s_present = s_onehot.sum(dim=0) > 0
        t_present = F.one_hot(t_sca_label_t, num_classes=CLASS_NUM).sum(dim=0) > 0
        common_class = (s_present & t_present).float()
        class_count = common_class.sum()

        if class_count.item() == 0:
            zero = torch.zeros(1, dtype=torch.float32, device=device)
            return zero, zero, zero

        s_vec_label_t = s_vec_label_t * common_class.view(1, CLASS_NUM)
        t_vec_label_t = t_vec_label_t * common_class.view(1, CLASS_NUM)

        weight_ss = torch.mm(s_vec_label_t, s_vec_label_t.t()) / class_count
        weight_tt = torch.mm(t_vec_label_t, t_vec_label_t.t()) / class_count
        weight_st = torch.mm(s_vec_label_t, t_vec_label_t.t()) / class_count

        return weight_ss.float(), weight_tt.float(), weight_st.float()

        # # label_list = list(set(s_label.data.cpu().numpy()))
        # #
        # # CLASS_NUM = len(label_list)
        # # print('label list', label_list)
        # # print('class num',CLASS_NUM)
        #
        # CLASS_NUM = int(torch.max(s_label)) + 1

        #计算核函数前的权值（源域）
        s_sca_label = s_label.cpu().data.numpy()

        s_vec_label = convert_to_onehot(s_sca_label,CLASS_NUM)
        s_sum = np.sum(s_vec_label, axis=0).reshape(1, CLASS_NUM)
        s_sum[s_sum == 0] = 100
        s_vec_label = s_vec_label / s_sum
        #计算核函数前的权值（目标域）
        t_sca_label = t_label.cpu().data.max(1)[1].numpy()
        #t_vec_label = convert_to_onehot(t_sca_label)

        t_vec_label = t_label.cpu().data.numpy()

        t_sum = np.sum(t_vec_label, axis=0).reshape(1, CLASS_NUM)

        t_sum[t_sum == 0] = 100
        t_vec_label = t_vec_label / t_sum

        weight_ss = np.zeros((batch_size, batch_size))
        weight_tt = np.zeros((batch_size, batch_size))
        weight_st = np.zeros((batch_size, batch_size))

        set_s = set(s_sca_label)
        set_t = set(t_sca_label)
        count = 0
        for i in range(CLASS_NUM):
            if i in set_s and i in set_t:
                s_tvec = s_vec_label[:, i].reshape(batch_size, -1)
                t_tvec = t_vec_label[:, i].reshape(batch_size, -1)
                ss = np.dot(s_tvec, s_tvec.T)
                weight_ss = weight_ss + ss# / np.sum(s_tvec) / np.sum(s_tvec)
                tt = np.dot(t_tvec, t_tvec.T)
                weight_tt = weight_tt + tt# / np.sum(t_tvec) / np.sum(t_tvec)
                st = np.dot(s_tvec, t_tvec.T)
                weight_st = weight_st + st# / np.sum(s_tvec) / np.sum(t_tvec)
                count += 1

        length = count  # len( set_s ) * len( set_t )
        if length != 0:
            weight_ss = weight_ss / length
            weight_tt = weight_tt / length
            weight_st = weight_st / length
        else:
            weight_ss = np.array([0])
            weight_tt = np.array([0])
            weight_st = np.array([0])
        return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32')
