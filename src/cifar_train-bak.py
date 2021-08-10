# -*- coding: utf-8 -*-
"""
# @file name  : main.py
# @author     : zjgbz https://github.com/zjgbz
# @date       : 2021-02-27
# @brief      : 模型训练主代码
"""
import matplotlib
# matplotlib.use('agg')
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))
from tqdm import *
import pickle
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from models.resnet_cifar10 import resnet32
from models.vgg_cifar10 import VGG
from models.lenet import LeNet
from tools.my_dataset import CifarDataset, CifarLTDataset
from tools.my_loss import CBLoss, LabelSmoothLoss
from tools.my_lr_schedule import CosineWarmupLr
from tools.model_trainer import ModelTrainer
from tools.common_tools import setup_seed, show_confMat, plot_line, Logger, check_data_dir
from config.training_config import cfg
from tools.progressively_balance import ProgressiveSampler
from datetime import datetime

setup_seed(12345)  # 先固定随机种子

if __name__ == '__main__':
    # 设置路径
    train_dir = os.path.join(BASE_DIR, "..", "..", "data", "cifar10", "cifar10_train")
    valid_dir = os.path.join(BASE_DIR, "..", "..", "data", "cifar10", "cifar10_test")
    check_data_dir(train_dir)
    check_data_dir(valid_dir)
    output_dir = os.path.join(BASE_DIR, "..", "..", "results")
    # log 输出文件夹配置
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
    log_dir = os.path.join(BASE_DIR, "..", "..", "results", time_str)  # 根据config中的创建时间作为文件夹名
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    path_log = os.path.join(log_dir, "log.log")
    logger = Logger(path_log)
    logger = logger.init_logger()

    # ------------------------------------ step 1/5 : 加载数据------------------------------------
    # 构建MyDataset实例
    # train_data = CifarDataset(path_dir=train_dir, transform=cfg.transforms_train)
    # valid_data = CifarDataset(path_dir=valid_dir, transform=cfg.transforms_valid)
    train_data = CifarLTDataset(root_dir=train_dir, transform=cfg.transforms_train, isTrain=True)
    valid_data = CifarLTDataset(root_dir=valid_dir, transform=cfg.transforms_valid, isTrain=False)
    # 构建DataLoder
    if cfg.resample:
        class_sample_counts = train_data.nums_per_cls
        class_weights = torch.tensor(class_sample_counts, dtype=torch.float)    # 计算各类别权重
        train_targets = [int(info[1]) for info in train_data.img_info]
        samples_weights = class_weights[train_targets]                                # 计算各样本权重
        sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)

        train_loader = DataLoader(dataset=train_data, batch_size=cfg.train_bs, shuffle=False, num_workers=cfg.workers,
                                  sampler=sampler)
        labels = []
        for data in train_loader:
            _, label, _ = data
            labels.extend(label.tolist())
        from collections import Counter
        c = Counter(labels)
        logger.info("re-sampling: ", c)
    else:
        train_loader = DataLoader(dataset=train_data, batch_size=cfg.train_bs, shuffle=True, num_workers=cfg.workers)
    valid_loader = DataLoader(dataset=valid_data, batch_size=cfg.valid_bs, num_workers=cfg.workers)

    if cfg.pb:
        sampler_generator = ProgressiveSampler(train_data, cfg.max_epoch)

    # ------------------------------------ step 2/5 : 定义网络------------------------------------
    # model = LeNet()
    # model = VGG("VGG16", classes=10)
    model = resnet32()
    model.to(cfg.device)

    # ------------------------------------ step 3/5 : 定义损失函数和优化器 ------------------------------------
    loss_f = nn.CrossEntropyLoss().to(cfg.device)
    if cfg.label_smooth:
        loss_f = LabelSmoothLoss(cfg.label_smooth_eps)
    if cfg.CBLoss:
        loss_f_train = CBLoss(train_data.nums_per_cls, train_data.cls_num, loss_type="softmax", beta=.9999, gamma=1.)
        loss_f_valid = CBLoss(valid_data.nums_per_cls, valid_data.cls_num, loss_type="softmax", beta=.9999, gamma=1.)
    else:
        loss_f_train = loss_f_valid = loss_f

    optimizer = optim.SGD(model.parameters(), lr=cfg.lr_init, momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    if cfg.is_warmup:
        iter_per_epoch = len(train_loader)
        scheduler = CosineWarmupLr(optimizer, batches=iter_per_epoch, max_epochs=cfg.max_epoch,
                                   base_lr=cfg.lr_init, final_lr=cfg.lr_final,
                                   warmup_epochs=cfg.warmup_epochs, warmup_init_lr=cfg.lr_warmup_init)
    else:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=cfg.factor, milestones=cfg.milestones)

    # ------------------------------------ step 4/5 : 训练 --------------------------------------------------
    # 记录训练配置信息
    logger.info("cfg:\n{}\n loss_f:\n{}\n scheduler:\n{}\n optimizer:\n{}\n model:\n{}".format(
        cfg, loss_f_train, scheduler, optimizer, model))

    loss_rec = {"train": [], "valid": []}
    acc_rec = {"train": [], "valid": []}
    best_acc, best_epoch = 0, 0
    for epoch in range(cfg.max_epoch):
        if cfg.pb:
            sampler, _ = sampler_generator(epoch)
            train_loader = DataLoader(dataset=train_data, batch_size=cfg.train_bs, shuffle=False,
                                      num_workers=cfg.workers,
                                      sampler=sampler)
        # 喂数据，训练模型
        loss_train, acc_train, mat_train, path_error_train = ModelTrainer.train(
            train_loader, model, loss_f_train, optimizer, scheduler, epoch, cfg, logger)
        loss_valid, acc_valid, mat_valid, path_error_valid = ModelTrainer.valid(
            valid_loader, model, loss_f_valid, cfg)
        logger.info("Epoch[{:0>3}/{:0>3}] Train Acc: {:.2%} Valid Acc:{:.2%} Train loss:{:.4f} Valid loss:{:.4f} LR:{}". \
                    format(epoch + 1, cfg.max_epoch, acc_train, acc_valid, loss_train, loss_valid,
                           optimizer.param_groups[0]["lr"]))

        # 学习率更新
        if not cfg.is_warmup:
            scheduler.step()  # StepLR

        # 记录训练信息
        loss_rec["train"].append(loss_train), loss_rec["valid"].append(loss_valid)
        acc_rec["train"].append(acc_train), acc_rec["valid"].append(acc_valid)
        # 保存混淆矩阵图
        show_confMat(mat_train, train_data.names, "train", log_dir, epoch=epoch, verbose=epoch == cfg.max_epoch - 1)
        show_confMat(mat_valid, valid_data.names, "valid", log_dir, epoch=epoch, verbose=epoch == cfg.max_epoch - 1)
        # 保存loss曲线， acc曲线
        plt_x = np.arange(1, epoch + 2)
        plot_line(plt_x, loss_rec["train"], plt_x, loss_rec["valid"], mode="loss", out_dir=log_dir)
        plot_line(plt_x, acc_rec["train"], plt_x, acc_rec["valid"], mode="acc", out_dir=log_dir)
        # 保存模型
        if best_acc < acc_valid or epoch == cfg.max_epoch-1:

            best_epoch = epoch if best_acc < acc_valid else best_epoch
            best_acc = acc_valid if best_acc < acc_valid else best_acc
            checkpoint = {"model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch,
                          "best_acc": best_acc}
            pkl_name = "checkpoint_{}.pkl".format(epoch) if epoch == cfg.max_epoch-1 else "checkpoint_best.pkl"
            path_checkpoint = os.path.join(log_dir, pkl_name)
            torch.save(checkpoint, path_checkpoint)

            # 保存错误图片的路径
            err_ims_name = "error_imgs_{}.pkl".format(epoch) if epoch == cfg.max_epoch-1 else "error_imgs_best.pkl"
            path_err_imgs = os.path.join(log_dir, err_ims_name)
            error_info = {}
            error_info["train"] = path_error_train
            error_info["valid"] = path_error_valid
            pickle.dump(error_info, open(path_err_imgs, 'wb'))

    logger.info("{} done, best acc: {} in :{}".format(
        datetime.strftime(datetime.now(), '%m-%d_%H-%M'), best_acc, best_epoch))


