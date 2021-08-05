# -*- coding: utf-8 -*-
"""
# @file name  : model_trainer.py
# @author     : https://github.com/zjgbz
# @date       : 2020-08-03
# @brief      : 模型训练类
"""
import torch
import numpy as np
from collections import Counter
from tools.mixup import mixup_criterion, mixup_data
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

class ModelTrainer(object):

    @staticmethod
    def train(data_loader, model, loss_f, optimizer, scheduler, epoch_idx, device, cfg, logger):
        model.train()

        class_num = data_loader.dataset.cls_num
        conf_mat = np.zeros((class_num, class_num))
        loss_sigma = []
        loss_mean = 0
        acc_avg = 0
        path_error = []
        label_list = []
        pred_list = []
        for i, data in enumerate(data_loader):

            # _, labels = data
            inputs, labels, path_imgs = data
            label_list.extend(labels.tolist())

            inputs, labels = inputs.to(device), labels.to(device)

            # mixup
            if cfg.mixup:
                mixed_inputs, label_a, label_b, lam = mixup_data(inputs, labels, cfg.mixup_alpha, device)
                inputs = mixed_inputs

            # forward & backward
            outputs = model(inputs)
            optimizer.zero_grad()
            # loss 计算
            if cfg.mixup:
                loss = mixup_criterion(loss_f, outputs.cpu(), label_a.cpu(), label_b.cpu(), lam)
            else:
                loss = loss_f(outputs.cpu(), labels.cpu())
            loss.backward()
            optimizer.step()

            # 统计loss
            loss_sigma.append(loss.item())
            loss_mean = np.mean(loss_sigma)
            #
            _, predicted = torch.max(outputs.data, 1)
            pred_list.extend(predicted.tolist())

            for j in range(len(labels)):
                cate_i = labels[j].cpu().numpy()
                pre_i = predicted[j].cpu().numpy()
                conf_mat[cate_i, pre_i] += 1.
                if cate_i != pre_i:
                    path_error.append((cate_i, pre_i, path_imgs[j]))    # 记录错误样本的信息
            acc_avg = conf_mat.trace() / conf_mat.sum()

            # AUROC
            labels_array = np.asarray(label_list)
            pred_array = np.asarray(pred_list)
            # auroc = roc_auc_score(labels_array, pred_array)
            prec_macro, recall_macro, F1_macro, _ = precision_recall_fscore_support(labels_array, pred_array, average = 'macro', zero_division = 0)
            prec_micro, recall_micro, F1_micro, _ = precision_recall_fscore_support(labels_array, pred_array, average = 'micro', zero_division = 0)

            # 每10个iteration 打印一次训练信息
            if i % cfg.log_interval == cfg.log_interval - 1:
                # logger.info(
                #     (f"Training: Epoch[{epoch_idx + 1:0>3}/{cfg.max_epoch:0>3}] Iteration[{i + 1:0>3}/{len(data_loader):0>3}] "
                #      f"Loss: {loss_mean:.4f} Acc:{acc_avg:.2%} Precision(macro):{prec_macro:.4f} Recall(macro):{recall_macro:.4f} "
                #      f"F1(macro):{F1_macro:.4f} Precision(micro):{prec_micro:.4f} Recall(micro):{recall_micro:.4f} F1(micro):{F1_micro:.4f}")
                # )
                logger.info(
                    (f"Training: Epoch[{epoch_idx + 1:0>3}/{cfg.max_epoch:0>3}] Iteration[{i + 1:0>3}/{len(data_loader):0>3}] "
                     f"Loss: {loss_mean:.4f} Acc:{acc_avg:.2%} F1(macro):{F1_macro:.4f}")
                )
        logger.info("epoch:{} sampler: {}".format(epoch_idx, Counter(label_list)))
        return loss_mean, acc_avg, prec_macro, recall_macro, F1_macro, prec_micro, recall_micro, F1_micro, conf_mat, path_error
        # return loss_mean, acc_avg, conf_mat, path_error

    @staticmethod
    def valid(data_loader, model, loss_f, device):
        model.eval()

        class_num = data_loader.dataset.cls_num
        conf_mat = np.zeros((class_num, class_num))
        loss_sigma = []
        path_error = []

        label_list = []
        pred_list = []
        for i, data in enumerate(data_loader):
            with torch.no_grad():
                inputs, labels, path_imgs = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = loss_f(outputs.cpu(), labels.cpu())

            # 统计混淆矩阵
            _, predicted = torch.max(outputs.data, 1)
            
            for j in range(len(labels)):
                cate_i = labels[j].cpu().numpy()
                pre_i = predicted[j].cpu().numpy()
                conf_mat[cate_i, pre_i] += 1.
                if cate_i != pre_i:
                    path_error.append((cate_i, pre_i, path_imgs[j]))   # 记录错误样本的信息

            # 统计loss
            loss_sigma.append(loss.item())

            label_list.extend(labels.tolist())
            pred_list.extend(predicted.tolist())
            # print(f"batch {i} completed.")

        acc_avg = conf_mat.trace() / conf_mat.sum()

        # AUROC
        # print(len(label_list), len(pred_list))
        labels_array = np.asarray(label_list)
        pred_array = np.asarray(pred_list)
        # auroc = roc_auc_score(labels_array, pred_array)

        prec_macro, recall_macro, F1_macro, _ = precision_recall_fscore_support(labels_array, pred_array, average = 'macro', zero_division = 0)
        prec_micro, recall_micro, F1_micro, _ = precision_recall_fscore_support(labels_array, pred_array, average = 'micro', zero_division = 0)

        return np.mean(loss_sigma), acc_avg, prec_macro, recall_macro, F1_macro, prec_micro, recall_micro, F1_micro, conf_mat, path_error