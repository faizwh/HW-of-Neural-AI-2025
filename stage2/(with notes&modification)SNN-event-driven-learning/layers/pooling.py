import torch
import torch.nn as nn
import torch.nn.functional as f
import global_v as glv
from torch.cuda.amp import custom_fwd, custom_bwd
from layers.functions import readConfig


class PoolLayer(nn.Module):
    def __init__(self, network_config, config, name):
        super(PoolLayer, self).__init__()
        self.name = name
        self.layer_config = config
        self.network_config = network_config
        self.type = config['type']
        kernel_size = config['kernel_size']

        # 到了这里终于知道为什么要使用readConfig把单个int变成(int, int)的形式了
        # 是为了得到一个核的格式
        self.kernel = readConfig(kernel_size, 'kernelSize')
        # self.in_shape = in_shape
        # self.out_shape = [in_shape[0], int(in_shape[1] / kernel[0]), int(in_shape[2] / kernel[1])]
        print('pooling')
        # print(self.in_shape)
        # print(self.out_shape)
        print("-----------------------------------------")

    def forward(self, x):
        pool_type = glv.network_config['pooling_type']
        assert(pool_type in ['avg', 'max', 'adjusted_avg'])
        T, n_batch, C, H, W = x.shape
        x = x.reshape(T * n_batch, C, H, W)
        if pool_type == 'avg':
            x = f.avg_pool2d(x, self.kernel)
        elif pool_type == 'max':
            x = f.max_pool2d(x, self.kernel)
        elif pool_type == 'adjusted_avg':
            # 论文中提到的adujsted average pooling
            x = PoolFunc.apply(x, self.kernel)
        x = x.reshape(T, n_batch, *x.shape[1:])
        return x

    def get_parameters(self):
        return

    def forward_pass(self, x, epoch):
        y1 = self.forward(x)
        return y1

    def weight_clipper(self):
        return

class PoolFunc(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, inputs, kernel):
        # avgpool2d 接收输入的格式是 (N, C, H, W)，在H,W两个维度上按照kernel进行池化
        outputs = f.avg_pool2d(inputs, kernel)
        ctx.save_for_backward(outputs, torch.tensor(inputs.shape), torch.tensor(kernel))
        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_delta):
        (outputs, input_shape, kernel) = ctx.saved_tensors
        kernel = kernel.tolist()
        # 为什么做这一步，因为output是kernel里面的数值的平均值，但是不希望根据kernel的大小来，因为考虑到有些输入的值基本就是0
        # 按照论文，这里默认池化层接收到的输入一定是脉冲形式的，形状为T, n_batch, C, H, W的
        outputs = 1 / outputs
        # 把取倒数后异常大的置为0，因为这一个位置对应的原本input的kernel内就全是0，若不是0，理论上无法超过kernel[0]*kernel[1]，防止梯度消失
        outputs[outputs > kernel[0] * kernel[1] + 1] = 0 
        # 此处为什么又除以一个kernel_size呢，因为output = sum(inputs) / (kernel[0] * kernel[1])
        # 所以1/outputs = (kernel[0] * kernel[1]) / sum(inputs)
        # 但是我们希望得到的是1/sum(inputs)，所以还需要再除以kernel_size
        outputs /= kernel[0] * kernel[1]
        # input_shape.tolist()[2:] = [H_in, W_in]
        grad = f.interpolate(grad_delta * outputs, size=input_shape.tolist()[2:])
        return grad, None