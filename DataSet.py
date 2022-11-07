import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import os
import csv


################################################################
# Build Data from CSV of StarCCM+
###############################################################

def dataset_cut(path_in, path_out):
    data_dir = os.listdir(path_in)
    for i in data_dir:
        location_in = os.path.abspath(path_in)
        in_fo = os.path.join(location_in, i)
        print(in_fo, "Begin to cut Data")
        df = pd.DataFrame(pd.read_csv(in_fo, index_col=0, usecols=[0, 1, 2, 3]))
        location_out = os.path.abspath(path_out)
        out_fo = os.path.join(location_out, i)
        df.to_csv(out_fo, encoding='utf-8')
        print(in_fo, "Complete cut Data")


def label_data(path_in, path_out):
    data_dir = os.listdir(path_in)
    for i in data_dir:
        location_in = os.path.abspath(path_in)
        in_fo = os.path.join(location_in, i)
        print(in_fo, "Begin to select label")
        df = pd.DataFrame(pd.read_csv(in_fo, index_col=0, usecols=[0, 1, 2, 3], nrows=1))
        location_out = os.path.abspath(path_out)
        out_fo = os.path.join(location_out, i)
        df.to_csv(out_fo, encoding='utf-8')
        print(in_fo, "Complete select label")


################################################################
# load data and data set up
###############################################################

class VelocityDataSet(Dataset):
    def __init__(self, data_dir, label_dir, transform=None):
        """
        :param data_dir: 数据文件路径
        :param label_dir: 标签文件路径
        :param transform: transform操作
        """
        self.transform = transform
        # 读文件夹下每个数据文件名称
        # os.listdir读取文件夹内的文件名称
        self.file_name = os.listdir(data_dir)
        # 读标签文件夹下的数据名称
        self.label_name = os.listdir(label_dir)

        self.data_path = []
        self.label_path = []

        # 让每一个文件的路径拼接起来
        for index in range(len(self.file_name)):
            self.data_path.append(os.path.join(data_dir, self.file_name[index]))
            self.label_path.append(os.path.join(label_dir, self.label_name[index]))

    def __len__(self):
        # 返回数据集长度
        return len(self.file_name)

    def __getitem__(self, index):
        # 获取每一个数据

        # 读取数据
        data = pd.read_csv(self.data_path[index], header=None)
        # 读取标签
        label = pd.read_csv(self.label_path[index], header=None)

        if self.transform:
            data = self.transform(data)
            label = self.transform(label)

        # 转成张量
        data = torch.tensor(data.values)
        label = torch.tensor(label.values)

        return data, label  # 返回数据和标签


if __name__ == '__main__':
    all_data_location = "E:/1"
    new_data_location = "E:/1-data"
    new_label_location = "E:/1-label"
    dataset_cut(all_data_location, new_data_location)
    label_data(all_data_location, new_label_location)

# data_dir = r"E:/DatasetTry1/Data/"
# label_dir = r"E:/DatasetTry1/label/"
# # 读取数据集
# train_dataset = VelocityDataSet(
#     data_dir=data_dir,
#     label_dir=label_dir,)
# # 加载数据集
# train_iter = DataLoader(train_dataset)
# file_name = "E:/DatasetTry1/XYZ_内部表_table_1.000000e+00.csv"
# dataset = pd.read_csv(file_name)
# # print(dataset.shape)
# data_all = dataset.values
# data_all = data_all.astype(float)
# data_all = torch.from_numpy(data_all)
# print(data_all)
# print(data_all.shape)
# velocity_magnitude = data_all[:, 0]
# velocity_i = data_all[:, 1]
# velocity_j = data_all[:, 2]
# velocity_k = data_all[:, 3]
