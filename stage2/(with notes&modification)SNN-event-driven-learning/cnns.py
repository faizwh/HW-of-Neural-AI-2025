import torch
import torch.nn as nn
import layers.conv as conv
import layers.pooling as pooling
import layers.dropout as dropout
import layers.linear as linear
from math import *
import global_v as glv


class Network(nn.Module):
    def __init__(self, input_shape=None):
        super(Network, self).__init__()
        self.layers = []
        # 改换一个新名字，不必重复添加glv的前缀，并不是全新的变量，而仍然是预设的全局变量
        # 创建全新的等值变量需要copy
        network_config, layers_config = glv.network_config, glv.layers_config
        print("Network Structure:")
        for key in layers_config:
            # 由key索引出config
            c = layers_config[key]
            # 实际上network_config并没有被用到conv.ConvLayer,linear.LinearLayer,pooling.PoolLayer的初始化当中
            if c['type'] == 'conv':
                self.layers.append(conv.ConvLayer(network_config, c, key))
            elif c['type'] == 'linear':
                self.layers.append(linear.LinearLayer(network_config, c, key))
            elif c['type'] == 'pooling':
                self.layers.append(pooling.PoolLayer(network_config, c, key))
            elif c['type'] == 'dropout':
                self.layers.append(dropout.DropoutLayer(c, key))
            else:
                raise Exception('Undefined layer type. It is: {}'.format(c['type']))

        #由layers_config创建网络的结构
        self.net = nn.Sequential(*self.layers)
        print("-----------------------------------------")

    def forward(self, inputs, labels, epoch, is_train):
        assert(is_train or labels==None)
        # spikes = f.psp(spike_input, self.network_config)
        spikes = inputs
        
        # 逐层处理，强制要求最后一层为线性层
        for i, l in enumerate(self.layers):
            if l.type == "dropout":
                if is_train:
                    spikes = l(spikes)
            elif i == len(self.layers) - 1:
                assert(l.type == 'linear')
                spikes = l.forward(spikes, labels)
            else:
                spikes = l.forward(spikes)

        return spikes

    def weight_clipper(self):
        for l in self.layers:
            l.weight_clipper()

    def train(self):
        for l in self.layers:
            l.train()

    def eval(self):
        for l in self.layers:
            l.eval()
