import pandas as pd
import torch
from torch.utils.data import DataLoader


class GetLoader(torch.utils.data.Dataset):
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label

    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels

    def __len__(self):
        return len(self.data)


file_name = "E:/DatasetTry1/XYZ_内部表_table_1.000000e+00.csv"
dataset = pd.read_csv(file_name)
# print(dataset.shape)
data_all = dataset.values
data_all = data_all.astype(float)
data_all = torch.from_numpy(data_all)
print(data_all)
print(data_all.shape)
velocity_magnitude = data_all[:, 0]
velocity_i = data_all[:, 1]
velocity_j = data_all[:, 2]
velocity_k = data_all[:, 3]
