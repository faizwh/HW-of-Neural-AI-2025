import math
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from os import listdir
from os.path import isfile, join


class NMNIST(Dataset):
    # N-Mnist(Neuromorphic-MNIST) dataset
    # spike形式的mnist数据集
    def __init__(self, dataset_path, T, transform=None):
        self.path = dataset_path
        self.samples = []
        self.labels = []
        self.transform = transform
        self.T = T
        for i in tqdm(range(10)):
            # mnist总共10个类别，每个类别的文件夹路径
            # 那看来要使用这个就得自己去预先下载n-mnist数据集了
            # 而且这个数据集按类别列出10个文件夹
            # tqdm显示进度条，本质是for i in range(10)
            # 这样把所有数据重新整合，按索引访问其内容和标签
            sample_dir = dataset_path + '/' + str(i) + '/' 
            for f in listdir(sample_dir):
                filename = join(sample_dir, f)
                if isfile(filename):
                    self.samples.append(filename)
                    self.labels.append(i)

    def __getitem__(self, index):
        filename = self.samples[index]
        label = self.labels[index]
        # NMNIST 数据集记录的是动态视觉事件（由事件相机拍摄），事件分为两种类型（data形状里的2的来历）：
        # ON 事件：像素亮度突然增加时触发；
        # OFF 事件：像素亮度突然降低时触发。
        data = np.zeros((2, 34, 34, self.T))

        f = open(filename, 'r')
        lines = f.readlines()
        for line in lines:
            if line is None:
                break
            line = line.split()
            line = [int(l) for l in line]
            pos = line[0] - 1
            if pos >= 1156:
                channel = 1 # 位置在第二个大通道里(>=34*34)
                pos -= 1156
            else:
                channel = 0
            y = pos % 34
            x = int(math.floor(pos/34))
            for i in range(1, len(line)):
                # # 处理时间点（每行从第2个元素开始是事件发生的时间步）
                if line[i] >= self.T:
                    break
                data[channel, x, y, line[i]-1] = 1
        if self.transform:
            data = self.transform(data)
            data = data.type(torch.float32)
        else:
            data = torch.FloatTensor(data)

        # Input spikes are reshaped to ignore the spatial dimension and the neurons are placed in channel dimension.
        # The spatial dimension can be maintained and used as it is.
        # It requires different definition of the dense layer.
        # 转为[T, on/off, 34, 34]的形式
        return data.permute(3,0,1,2), label

    def __len__(self):
        return len(self.samples)


def get_nmnist(data_path, network_config):
    T = network_config['n_steps']
    print("loading NMNIST")
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    train_path = data_path + '/Train'
    test_path = data_path + '/Test'
    
    trainset = NMNIST(train_path, T)
    testset = NMNIST(test_path, T)

    return trainset, testset
