# -*- coding: utf-8 -*-
"""
# @file name  : cell_painting_lincs.py
# @author     : https://github.com/zjgbz
# @date       : 2021-08-02
# @brief      : LINCS Cell Painting数据集读取
"""

import os
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd

class CellMorphoDataset(Dataset):
    cls_num = 2

    def __init__(self, img_dict_df, img_dir_col, label_col, transform=None):
        """
        获取数据集的路径、预处理的方法
        """
        self.img_dict_df = img_dict_df
        self.transform = transform
        self.img_info = []   # [(path, label), ... , ]
        self.label_array = None
        self._get_img_info()
        self.img_dir_col = img_dir_col
        self.label_col = label_col

    def __getitem__(self, index):
        """
        输入标量index, 从硬盘中读取数据，并预处理，to Tensor
        :param index:
        :return:
        """
        path_img, label = self.img_info[index]
        image_5D = np.load(path_img) # (5, 520, 696)
        if self.transform is not None:
            image_5D = image_5D.transpose(1, 2, 0)
            augmented = self.transform(image = image_5D)
            img = augmented['image']
            img = img / 15
        
        return img, label, path_img

    def __len__(self):
        """
        返回数据集的长度
        :return:
        """
        if len(self.img_info) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(
                self.root_dir))   # 代码具有友好的提示功能，便于debug
        return len(self.img_info)

    def _get_img_info(self):
        """
        实现数据集的读取，将硬盘中的数据路径，标签读取进来，存在一个list中
        path, label
        :return:
        """
        path_imgs = img_dict_df.loc[:, img_dir_col].values.tolist()

        # read labels from pandas dataframe -- "assay_id_737823"
        label_array = img_dict_df.loc[:, label_col].values.tolist()
        self.label_array = label_array

        # match the path_imgs and label_array
        self.img_info = [(p, label) for p, label in zip(path_imgs, label_array)]

if __name__ == "__main__":

    img_dict_dir = "/gxr/minzhi/multiomics_data/E_Metadata/raw"
    img_dict_filename = "assay_id_737823_raw_img_unique_train_val.csv"
    img_dict_dir_filename = os.path.join(img_dict_dir, img_dict_filename)
    img_dict_df = pd.read_csv(img_dict_dir_filename, sep = ",", header = 0, index_col = None)

    img_dir_col = "npy_path"
    label_col = "assay_id_737823"
    test_dataset = CellMorphoDataset(img_dict_df, img_dir_col, label_col)

    print(len(test_dataset))
    print(next(iter(test_dataset)))