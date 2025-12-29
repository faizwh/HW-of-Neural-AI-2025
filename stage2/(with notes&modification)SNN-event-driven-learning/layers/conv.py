import torch
import torch.nn as nn
import torch.nn.functional as f
from layers.functions import neuron_forward, neuron_backward, bn_forward, bn_backward, readConfig, initialize
import global_v as glv
import torch.backends.cudnn as cudnn
from torch.utils.cpp_extension import load_inline, load
from torch.cuda.amp import custom_fwd, custom_bwd
from datetime import datetime

cpp_wrapper = load(name="cpp_wrapper", sources=["layers/cpp_wrapper.cpp"], verbose=True)


class ConvLayer(nn.Conv2d):
    def __init__(self, network_config, config, name, groups=1):
        self.name = name
        # threshold = 1
        self.threshold = config['threshold'] if 'threshold' in config else None
        self.type = config['type']
        in_features = config['in_channels']
        out_features = config['out_channels']
        kernel_size = config['kernel_size'] # kernel_size = 3

        padding = config['padding'] if 'padding' in config else 0
        stride = config['stride'] if 'stride' in config else 1
        dilation = config['dilation'] if 'dilation' in config else 1

        self.kernel = readConfig(kernel_size, 'kernelSize') # = (kernel_size, kernel_size)
        self.stride = readConfig(stride, 'stride') # 等等，同上
        # self.padding = readConfig(padding, 'stride')
        # self.dilation = readConfig(dilation, 'stride')
        # 微不足道的小纰漏
        self.padding = readConfig(padding, 'padding')
        self.dilation = readConfig(dilation, 'dilation')

        # 初始化父类nn.conv2d
        super(ConvLayer, self).__init__(in_features, out_features, self.kernel, self.stride, self.padding,
                                        self.dilation, groups, bias=False)
        # weight是父类创建的参数
        self.weight = torch.nn.Parameter(self.weight.cuda(), requires_grad=True)
        # 这俩似乎就不是了
        self.norm_weight = torch.nn.Parameter(torch.ones(out_features, 1, 1, 1, device='cuda'))
        self.norm_bias = torch.nn.Parameter(torch.zeros(out_features, 1, 1, 1, device='cuda'))

        print('conv')
        print(f'Shape of weight is {list(self.weight.shape)}')  # Cout * Cin * Hk * Wk
        print(f'stride = {self.stride}, padding = {self.padding}, dilation = {self.dilation}, groups = {self.groups}')
        print("-----------------------------------------")

    def forward(self, x):
        if glv.init_flag:
            glv.init_flag = False
            x = initialize(self, x)
            glv.init_flag = True
            return x

        # self.weight_clipper()
        config_n = glv.network_config
        theta_m = 1 / config_n['tau_m']
        theta_s = 1 / config_n['tau_s']
        theta_grad = 1 / config_n['tau_grad'] if config_n[
                                                     'gradient_type'] == 'exponential' else -123456789  # instead of None
        # 又是重写的函数
        y = ConvFunc.apply(x, self.weight, self.norm_weight, self.norm_bias,
                           (self.bias, self.stride, self.padding, self.dilation, self.groups),
                           (theta_m, theta_s, theta_grad, self.threshold))
        return y

    def weight_clipper(self):
        w = self.weight.data
        w = w.clamp(-4, 4)
        self.weight.data = w


class ConvFunc(torch.autograd.Function):
    @staticmethod
    # 标记一个方法为静态方法，即该方法属于类本身，不需要依赖类的实例（self）或类的状态（cls），可以直接通过类名调用。
    @custom_fwd
    # 用于标记自定义求导函数的反向传播方法
    def forward(ctx, inputs, weight, norm_weight, norm_bias, conv_config, neuron_config):
        # input.shape: T * n_batch * C_in * H_in * W_in
        bias, stride, padding, dilation, groups = conv_config
        T, n_batch, C, H, W = inputs.shape
        # bn_forward在functions.py中
        inputs, mean, var, weight_ = bn_forward(inputs, weight, norm_weight, norm_bias)

        # 使用归一化的weight_进行卷积操作
        in_I = f.conv2d(inputs.reshape(T * n_batch, C, H, W), weight_, bias, stride, padding, dilation, groups)
        _, C, H, W = in_I.shape
        in_I = in_I.reshape(T, n_batch, C, H, W)

        # 脉冲神经元发放脉冲
        # delta_u:膜电位变化量
        # delta_u_t:膜电位变化量的时间导数
        # outputs:脉冲输出
        delta_u, delta_u_t, outputs = neuron_forward(in_I, neuron_config)

        ctx.save_for_backward(delta_u, delta_u_t, inputs, outputs, weight, norm_weight, norm_bias, mean, var)
        ctx.conv_config = conv_config

        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_delta):
        # shape of grad_delta: T * n_batch * C * H * W
        (delta_u, delta_u_t, inputs, outputs, weight, norm_weight, norm_bias, mean, var) = ctx.saved_tensors
        bias, stride, padding, dilation, groups = ctx.conv_config
        # 梯度仅回传到脉冲输出上
        grad_delta *= outputs
        
        # sum_next = grad_delta.sum().item()
        # print("Max of dLdt: ", abs(grad_delta).max().item())

        grad_in_, grad_w_ = neuron_backward(grad_delta, outputs, delta_u, delta_u_t)
        weight_ = (weight - mean) / torch.sqrt(var + 1e-5) * norm_weight + norm_bias

        T, n_batch, C, H, W = grad_delta.shape
        inputs = inputs.reshape(T * n_batch, *inputs.shape[2:])
        grad_in_, grad_w_ = map(lambda x: x.reshape(T * n_batch, C, H, W), [grad_in_, grad_w_])
        grad_input = cpp_wrapper.cudnn_convolution_backward_input(inputs.shape, grad_in_.to(weight_), weight_, padding,
                                                                  stride, dilation, groups,
                                                                  cudnn.benchmark, cudnn.deterministic,
                                                                  cudnn.allow_tf32) * inputs
        grad_weight = cpp_wrapper.cudnn_convolution_backward_weight(weight.shape, grad_w_.to(inputs), inputs, padding,
                                                                    stride, dilation, groups,
                                                                    cudnn.benchmark, cudnn.deterministic,
                                                                    cudnn.allow_tf32)

        grad_weight, grad_bn_w, grad_bn_b = bn_backward(grad_weight, weight, norm_weight, norm_bias, mean, var)

        # sum_last = grad_input.sum().item()
        # print(f'sum_next = {sum_next}, sum_last = {sum_last}')
        # assert(abs(sum_next - sum_last) < 1)
        return grad_input.reshape(T, n_batch, *inputs.shape[1:]) * 0.85, grad_weight, grad_bn_w, grad_bn_b, None, None, None
