import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import PIL
import math
import numpy as np


class packaging_class(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.transform = transform
        self.dataset = dataset

    def __getitem__(self, index):
        data, label = self.dataset[index]
        data = torch.FloatTensor(data)
        if self.transform:
            data = self.transform(data)
        # 就是在取出对应数据的时候进行一下类内存储的变换
        return data, label

    def __len__(self):
        return len(self.dataset)


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h = img.size(2)
        w = img.size(3)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)
        # 随机挖掉这么一个（至多）长宽均为length的正方形区域
        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img


def function_nda(data, M=1, N=2):
    c = 15 * N # 旋转角度范围（-c到+c度）
    rotate_tf = transforms.RandomRotation(degrees=c)
    e = 8 * N # 遮挡区域边长范围（至多到e）
    cutout_tf = Cutout(length=e)

    def roll(data, N=1):
        a = N * 2 + 1
        off1 = np.random.randint(-a, a + 1)
        off2 = np.random.randint(-a, a + 1)
        # 数据整体按[off1, off2]进行滚动，而不是偏移
        # 在H维度上，向下滚动off1个单位，超出H的自动补位到前面空缺的位置；
        # 在W维度上，向右滚动off2个单位
        # 所以叫做滚动而不是偏移
        return torch.roll(data, shifts=(off1, off2), dims=(2, 3))

    def rotate(data, N):
        return rotate_tf(data)

    def cutout(data, N):
        return cutout_tf(data)

    transforms_list = [roll, rotate, cutout]
    # 随机选取M个变换进行操作
    sampled_ops = np.random.choice(transforms_list, M)
    for op in sampled_ops:
        data = op(data, N)
    return data


def TTFS(data, T):
    # data: C*H*W
    # output: T*C*H*W
    C, H, W = data.shape
    low, high = torch.min(data), torch.max(data)
    data = ((data - low) / (high - low) * T).long() 
    # 归一化并转为整数
    # data中每个位置的值现在代表该位置发射脉冲的时间步
    # T --> T-1
    data = torch.clip(data, 0, T - 1)
    res = torch.zeros(T, C, H, W, device=data.device)
    # data.unsqueeze(0)-> 1*C*H*W
    # torch.Tensor.scatter_(dim, index, src) 是一个按索引填充值的函数
    # 在 dim 维度上，将 src 的值填充到 res 中 index 指定的位置
    # index里的值代表在dim上的取值
    # 即res[index[i][j],j] = src[i][j] 此处j可以是一个tuple，位置的复合
    return res.scatter_(0, data.unsqueeze(0), 1)
