import torch
import torch.nn as nn
import layers.conv as conv
import layers.pooling as pooling
import layers.dropout as dropout
import layers.linear as linear
from torch.cuda.amp import custom_fwd, custom_bwd
import global_v as glv


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    config = {'in_channels': in_planes, 'out_channels': out_planes, 'type': 'conv',
              'kernel_size': 3, 'padding': 1, 'stride': stride, 'dilation': dilation, 'threshold': 1}
    # dilation是“膨胀”，是将卷积核放大，在卷积核元素之间插入dilation-1个0，把k*k的卷积核放大为(k-1+dilation)*(k-1+dilation)
    return conv.ConvLayer(network_config=None, config=config, name=None)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    config = {'in_channels': in_planes, 'out_channels': out_planes, 'type': 'conv',
              'kernel_size': 1, 'padding': 0, 'stride': stride, 'threshold': 1}
    return conv.ConvLayer(network_config=None, config=config, name=None)


class BasicBlock(nn.Module):
    expansion = 1 # 通道扩展程度——尽管没有使用到
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, **kwargs):
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            # 通过downsample将原本的input x 变成可以和out相加的模样
            identity = self.downsample(x)

        # may need custom backward
        # out = out + identity
        out = AddFunc.apply(out, identity)

        return out

# 从此开始是需要和梯度有些关系的了，自己设计梯度，自己存储需要的变量
class AddFunc(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a + b

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        a, b = ctx.saved_tensors
        s = a + b
        s[s == 0] = 1
        # 这里梯度被放缩了，按照贡献比例做分配
        # 似乎是为了维持论文中想要证明的梯度总和不变性，以及减小无效路径带来的干扰
        # 但是论文中没有专门提出来为了resnet做出的这样的适配
        return grad * a / s, grad * b / s


class SpikingResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, groups=1, width_per_group=64, norm_layer=None, **kwargs):
        # layers：每个残差层包含的残差块数量（如[2,2,2,2]表示 4 个残差层，各含 2 个块）
        # Block 可以填入 BasicBlock等等参数
        # [2,2,2,2]等等具体来源参见下面的子类Network
        super(SpikingResNet, self).__init__()
        self._norm_layer = norm_layer

        self.inplanes = 128
        self.dilation = 1

        self.groups = groups
        self.base_width = width_per_group

        # config only for conv
        config = {'in_channels': 3, 'out_channels': self.inplanes, 'type': 'conv',
                  'kernel_size': 5, 'padding': 2, 'stride': 1, 'dilation': 1, 'threshold': 1}
        # 处理原始图像
        self.conv1 = conv.ConvLayer(network_config=None, config=config, name=None)

        # self.maxpool = layer.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 128, layers[0], stride=2, **kwargs)
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2, **kwargs)
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2, **kwargs)
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=2, **kwargs)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=1, **kwargs)
        config = {'type': 'pool', 'kernel_size': 32 // 2 ** 3}
        self.pool = pooling.PoolLayer(network_config=None, config=config, name=None)
        config = {'type': 'linear', 'n_inputs': 512 * block.expansion, 'n_outputs': num_classes, 'threshold': 1}
        # 最后一个全连接层首尾
        self.fc = linear.LinearLayer(network_config=None, config=config, name=None)

    def _make_layer(self, block, planes, blocks, stride=1, **kwargs):
        # 把planes理解为正常的channels
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if stride != 1 or self.inplanes != planes * block.expansion:
            # stride若是不等于1，那么从H*W的尺度上，就变小了，残差连接就很需要把x重新映射成可以和out相加的样子，
            # 自然而然就是downsample
            # self.inplanes != planes * block.expansion时，就是运行到目前make的layer前，x的通道数，和运行变换后的通道数不匹配
            # 这个时候其实本质是调整通道数，但是和downsample实际发生并不冲突，所以一并并入downsample当中就好
            # 归根结底有多downsample是从stride来看的
            downsample = conv1x1(self.inplanes, planes * block.expansion, stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, **kwargs))
        # 从这里可以看出，self.inplane记录的是经过makelayer后，当前应当的通道数
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation, **kwargs))

        return nn.Sequential(*layers)

    def forward(self, x, labels, epoch, is_train):
        assert (is_train or labels == None)
        # See note [TorchScript super()]
        x = self.conv1(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)

        x = self.pool(x)
        x = self.fc(x, labels)

        return x


class Network(SpikingResNet):
    def __init__(self, input_shape=None):
        super(Network, self).__init__(BasicBlock, [2, 2, 2, 2], glv.network_config['n_class'])
        print("-----------------------------------------")

