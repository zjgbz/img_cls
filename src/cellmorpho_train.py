# -*- coding: utf-8 -*-
"""
# @file name  : cellmorpho_train.py
# @author     : zjgbz https://github.com/zjgbz
# @date       : 2021-08-02
# @brief      : 模型训练主代码
"""

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))

import pandas as pd
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
from tools.model_trainer import ModelTrainer
from tools.common_tools import *
from tools.my_loss import LabelSmoothLoss
from config.cellmorpho_config import cfg
from datetime import datetime
from datasets.cell_painting_lincs import CellMorphoDataset

setup_seed(12345678)  # 先固定随机种子
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--lr', default=None, help='learning rate')
parser.add_argument('--bs', default=None, help='training batch size')
parser.add_argument('--max_epoch', default=None)
parser.add_argument('--img_dict_dir', default=r"/gxr/minzhi/multiomics_data/E_Metadata/raw",
                    help="path to your dataset")
parser.add_argument('--label_col', default="level_2", help="name of the clinical labels")
args = parser.parse_args()

cfg.lr_init = args.lr if args.lr else cfg.lr_init
cfg.train_bs = args.bs if args.bs else cfg.train_bs
cfg.max_epoch = args.max_epoch if args.max_epoch else cfg.max_epoch
cfg.label_col = args.label_col if args.label_col else cfg.label_col

if __name__ == "__main__":
    # step0: setting path
    train_dict_dir_filename = os.path.join(args.img_dict_dir, f"{cfg.clinical_type}-{cfg.label_col}_raw_img_dup_train_arti.csv")
    val_dict_dir_filename = os.path.join(args.img_dict_dir, f"{cfg.clinical_type}-{cfg.label_col}_raw_img_dup_val_arti.csv")
    check_data_dir(train_dict_dir_filename)
    check_data_dir(val_dict_dir_filename)
    train_dict_df = pd.read_csv(train_dict_dir_filename, sep = ",", header = 0, index_col = None)
    val_dict_df = pd.read_csv(val_dict_dir_filename, sep = ",", header = 0, index_col = None)

    # 创建logger
    res_dir = os.path.join(BASE_DIR, "..", "results")
    logger, log_dir = make_logger(res_dir)

    # step1： 数据集
    # 构建MyDataset实例， 构建DataLoder
    train_data = CellMorphoDataset(img_dict_df=train_dict_df, img_dir_col = cfg.img_dir_col, label_col = cfg.label_col, transform=cfg.transforms_train)
    valid_data = CellMorphoDataset(img_dict_df=val_dict_df, img_dir_col = cfg.img_dir_col, label_col = cfg.label_col, transform=cfg.transforms_valid)
    train_loader = DataLoader(dataset=train_data, batch_size=cfg.train_bs, shuffle=True, num_workers=cfg.workers)
    valid_loader = DataLoader(dataset=valid_data, batch_size=cfg.valid_bs, shuffle=True, num_workers=cfg.workers)

    # step2: 模型
    model = get_model(cfg, train_data.cls_num, logger)
    model.to(device)  # to device， cpu or gpu

    # step3: 损失函数、优化器
    if cfg.label_smooth:
        loss_f = LabelSmoothLoss(cfg.label_smooth_eps)
    else:
        loss_f = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr_init, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=cfg.factor, milestones=cfg.milestones)

    # step4: 迭代训练
    # 记录训练所采用的模型、损失函数、优化器、配置参数cfg
    logger.info("cfg:\n{}\n loss_f:\n{}\n scheduler:\n{}\n optimizer:\n{}\n model:\n{}".format(
        cfg, loss_f, scheduler, optimizer, model))

    loss_rec = {"train": [], "valid": []}
    F1_macro_rec = {"train": [], "valid": []}
    F1_micro_rec = {"train": [], "valid": []}
    acc_rec = {"train": [], "valid": []}
    best_F1_macro, best_F1_micro, best_acc, best_epoch = 0, 0, 0, 0

    for epoch in range(cfg.max_epoch):

        loss_train, acc_train, prec_macro_train, recall_macro_train, F1_macro_train, prec_micro_train, recall_micro_train, F1_micro_train, mat_train, path_error_train = ModelTrainer.train(
            train_loader, model, loss_f, optimizer, scheduler, epoch, device, cfg, logger)

        loss_valid, acc_valid, prec_macro_valid, recall_macro_valid, F1_macro_valid, prec_micro_valid, recall_micro_valid, F1_micro_valid, mat_valid, path_error_valid = ModelTrainer.valid(
            valid_loader, model, loss_f, device)

        # logger.info(
        #     (f"Epoch[{epoch + 1:0>3}/{cfg.max_epoch:0>3}] Train F1(macro): {F1_macro_train:.4f} Valid F1(macro):{F1_macro_valid:.4f} "
        #      f"Train F1(micro): {F1_micro_train:.4f} Valid F1(micro):{F1_micro_valid:.4f} "
        #      f"Train loss:{loss_train:.4f} Valid loss:{loss_valid:.4f} LR:{optimizer.param_groups[0]['lr']}")
        # )

        logger.info(
            (f"Epoch[{epoch + 1:0>3}/{cfg.max_epoch:0>3}] Train Acc: {acc_train:.4f} Valid Acc:{acc_valid:.4f} "
             f"Train loss:{loss_train:.4f} Valid loss:{loss_valid:.4f} LR:{optimizer.param_groups[0]['lr']}")
        )

        scheduler.step()

        # 记录训练信息
        loss_rec["train"].append(loss_train), loss_rec["valid"].append(loss_valid)
        F1_macro_rec["train"].append(F1_macro_train), F1_macro_rec["valid"].append(F1_macro_valid)
        F1_micro_rec["train"].append(F1_micro_train), F1_micro_rec["valid"].append(F1_micro_valid)
        acc_rec["train"].append(acc_train), acc_rec["valid"].append(acc_valid)

        # 保存loss曲线， acc曲线
        plt_x = np.arange(1, epoch + 2)
        plot_line(plt_x, loss_rec["train"], plt_x, loss_rec["valid"], mode="loss", out_dir=log_dir)
        plot_line(plt_x, F1_macro_rec["train"], plt_x, F1_macro_rec["valid"], mode="F1(macro)", out_dir=log_dir)
        plot_line(plt_x, F1_micro_rec["train"], plt_x, F1_micro_rec["valid"], mode="F1(micro)", out_dir=log_dir)
        plot_line(plt_x, acc_rec["train"], plt_x, acc_rec["valid"], mode="acc", out_dir=log_dir)

        # 模型保存
        if best_acc < acc_valid or epoch == cfg.max_epoch - 1:
            best_epoch = epoch if best_acc < acc_valid else best_epoch
            best_acc = acc_valid if best_acc < acc_valid else best_acc
            checkpoint = {"model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch,
                          "best_acc": best_acc}
            pkl_name = "checkpoint_{}.pkl".format(epoch) if epoch == cfg.max_epoch - 1 else "checkpoint_best.pkl"
            path_checkpoint = os.path.join(log_dir, pkl_name)
            torch.save(checkpoint, path_checkpoint)

            # 保存错误图片的路径
            err_ims_name = "error_imgs_{}.pkl".format(epoch) if epoch == cfg.max_epoch-1 else "error_imgs_best.pkl"
            path_err_imgs = os.path.join(log_dir, err_ims_name)
            error_info = {}
            error_info["train"] = path_error_train
            error_info["valid"] = path_error_valid
            pickle.dump(error_info, open(path_err_imgs, 'wb'))

        # 保存混淆矩阵图
        show_confMat(mat_train, train_data.names, "train", log_dir, epoch=epoch, verbose=epoch == best_epoch)
        show_confMat(mat_valid, valid_data.names, "valid", log_dir, epoch=epoch, verbose=epoch == best_epoch)

    logger.info("{} done, best Acc: {} in :{}".format(
        datetime.strftime(datetime.now(), '%m-%d_%H-%M'), best_acc, best_epoch))