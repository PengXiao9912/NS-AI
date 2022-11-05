import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import os
import time


################################################################
# Build Data from CSV of StarCCM+
###############################################################

# 读取csv数据第2-4列
def readwrite(inputfile, output_file):
    data = pd.read_csv(inputfile, sep=',', header=None)
    data.to_csv(output_file, sep=',', columns=[1, 2, 3, ], header=None, index=False)


# 计时函数
def getRunTimes(fun, input_file, output_file):
    begin_time = int(round(time.time() * 1000))
    fun(input_file, output_file)
    end_time = int(round(time.time() * 1000))
    print('Data processing completed')
    print("Data processing total time：", (end_time - begin_time), "ms")


# 调用
input_file = "E:/in.csv"
output_file = "E:/out.csv"
readwrite(input_file, output_file)
getRunTimes(readwrite, input_file, output_file)  # 使用dataframe读写数据


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
