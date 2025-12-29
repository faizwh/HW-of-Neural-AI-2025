import torch
import global_v as glv
from torch.utils.cpp_extension import load

try:
    neuron_cuda = load(name="neuron_cuda", sources=["layers/neuron_cuda.cpp", 'layers/neuron_cuda_kernel.cu'],
                       verbose=True)
except:
    print('Cannot load cuda neuron kernel.')


def readConfig(data, name):
    if type(data) == int:
        res = (data, data)
    else: # str
        try:
            assert(data[0] == '(' and data[-1] == ')')
            data = data[1:len(data)-1]
            x, y = map(int, data.split(','))
            res = (x, y)
        except:
            raise Exception(f'The format of {name} is illegal!')
    return res


"""
To solve this problem, we start with layers of arbitrarily initialized weights
and scale them by certain multiples, which can make the average firing rate to be a certain number
for each layer.
"""

def initialize(layer, spikes):
    avg_spike_init = glv.network_config['avg_spike_init'] # 在mnist.yaml里，avg_spike_init = 0.5
    from math import sqrt
    T = spikes.shape[0]
    # 后三分之一阶段的起始点
    t_start = T * 2 // 3

    low, high = 0.05, 500
    while high / low >= 1.01:
        mid = sqrt(high * low) # 几何平均
        layer.norm_weight.data *= mid
        outputs = layer.forward(spikes)
        layer.norm_weight.data /= mid
        n_neuron = outputs[0].numel()
        avg_spike = torch.sum(outputs[t_start:]) / n_neuron
        # 后三分之一的时间或许是“稳定期”
        if avg_spike > avg_spike_init / T * (T - t_start) * 1.2:
            high = mid
        # 二分法调整norm_weight.data, 使得该layer能够输出接近avg_spike_init / T * (T - t_start)脉冲量，并且留下1.2倍的脉冲数量
        else:
            low = mid
    layer.norm_weight.data *= mid
    return layer.forward(spikes)


def norm(inputs):
    # 这个函数似乎没有被使用过
    # 而且target_spike似乎多除了一个T
    T = inputs.shape[0]
    t_start = T * 2 // 3
    if (inputs >= 0).all():
        num_spike = (torch.sum(inputs[t_start:]) + 1e-5)
        target_spike = inputs.numel() / T * (T - t_start) / T
        inputs = inputs / num_spike * target_spike
    return inputs


def bn_forward(inputs, weight, norm_weight, norm_bias):
    # 作用的对象是weight，而不是输入x，有些怪异
    # inputs = norm(inputs)
    C = weight.shape[0]
    # print(weight.shape)
    mean, var = torch.mean(weight.reshape(C, -1), dim=1), torch.std(weight.reshape(C, -1), dim=1) ** 2
    shape = (-1, 1, 1, 1) if len(weight.shape) == 4 else (-1, 1)
    mean, var, norm_weight, norm_bias = [x.reshape(*shape) for x in [mean, var, norm_weight, norm_bias]]
    # norm_weight == weight of batch norm
    weight_ = (weight - mean) / torch.sqrt(var + 1e-5) * norm_weight + norm_bias
    return inputs, mean, var, weight_


def bn_backward(grad_weight, weight, norm_weight, norm_bias, mean, var):
    C = weight.shape[0]
    std_inv = 1 / torch.sqrt(var + 1e-5)
    shape = (-1, 1, 1, 1) if len(weight.shape) == 4 else (-1, 1)
    weight_ = (weight - mean) * std_inv * norm_weight.reshape(*shape) + norm_bias.reshape(*shape)
    # 输入的grad_weight实为上面bn_forward计算出的weight_的梯度（应该是）
    # 然后计算对于norm_bias和norm_weight的梯度，还有对于原来的weight的梯度
    grad_bn_b = torch.sum(grad_weight.reshape(C, -1), dim=1).reshape(norm_bias.shape)
    # 但是为什么是乘上weight_而不是(weight - mean) / torch.sqrt(var + 1e-5)，是不是计算错误了
    # 这样grad_bn_w不是多乘了一个norm_weight吗，应当是不准确的
    grad_bn_w = torch.sum((grad_weight * weight_).reshape(C, -1), dim=1).reshape(norm_weight.shape)
    grad_weight *= norm_weight.reshape(*shape)
    m = weight.numel() // C
    # 这里grad_var多除以一个m，是为了后续梯度计算方便吗
    # var = [(weight-mean)**2].sum()/m
    # grad_var初步计算时似乎也多除以了m
    # 还似乎没有进行.sum操作, grad_var和grad_weight应当有梯度上的sum，但是添加sum后需要进一步调整形状匹配后续计算
    grad_var = grad_weight * (weight - mean) / m * (-0.5) * std_inv ** 3
    grad_mean = -grad_weight * std_inv
    grad_weight = grad_weight * std_inv + grad_var * 2 * (weight - mean) / m + grad_mean / m
    return grad_weight, grad_bn_w, grad_bn_b

"""
# 尝试进行修正，将通过实际运行检验是否会产生巨大的影响，考虑到源代码最终确实训练成功了
def bn_backward(grad_weight, weight, norm_weight, norm_bias, mean, var):
    C = weight.shape[0]
    std_inv = 1 / torch.sqrt(var + 1e-5)
    shape = (-1, 1, 1, 1) if len(weight.shape) == 4 else (-1, 1)
    # # 修正1，使用weight_norm
    weight_norm = (weight - mean) * std_inv
    grad_bn_b = torch.sum(grad_weight.reshape(C, -1), dim=1).reshape(norm_bias.shape)
    grad_bn_w = torch.sum((grad_weight * weight_norm).reshape(C, -1), dim=1).reshape(norm_weight.shape)
    grad_weight *= norm_weight.reshape(*shape)
    m = weight.numel() // C
    # var = [(weight-mean)**2].sum()/m
    # # 修正2，除去疑似多除的m
    grad_var = grad_weight * (weight - mean)* (-0.5) * std_inv ** 3 
    grad_mean = -grad_weight * std_inv
    grad_weight = grad_weight * std_inv + grad_var * 2 * (weight - mean) / m + grad_mean / m
    return grad_weight, grad_bn_w, grad_bn_b

# 实际使用时，准确率上升快，抖动略明显，23轮后准确率稳定在99.90%以上，30轮后准确率在99.95%以上；测试集准确率最高来到99.49%
# 经过图像对比，实际证明梯度修正后能够为网络带来一定的提升，训练时曲线的稳定性有肉眼可见的改善，但是由于原本的代码也能够训练成功，所以提升并不是质变级别的
# 事实上修正前的梯度最终测试集上准确率为99.38%（训练集上为99.98%），修正后测试集上为99.47%（最高为99.50%，发生在训练中途）（训练集上为99.99%）
"""


@torch.jit.script
def neuron_forward_py(in_I, theta_m, theta_s, theta_grad, threshold, is_forward_leaky, is_grad_exp):
    # syn_m & syn_s: (1-theta_m)^t(approximation of exp(-theta_m) ) & (1-theta_s)^t(approximation of exp(-theta_s) ) in eps(t)
    # syn_grad: (1-theta_grad)^t in backward
    u_last = torch.zeros_like(in_I[0])
    syn_m, syn_s, syn_grad = torch.zeros_like(in_I[0]), torch.zeros_like(in_I[0]), torch.zeros_like(in_I[0])
    delta_u, delta_u_t, outputs = torch.zeros_like(in_I), torch.zeros_like(in_I), torch.zeros_like(in_I)
    T = in_I.shape[0]
    for t in range(T):
        syn_m = (syn_m + in_I[t]) * (1 - theta_m)
        syn_s = (syn_s + in_I[t]) * (1 - theta_s)
        syn_grad = (syn_grad + in_I[t]) * (1 - theta_grad)

        if not is_forward_leaky: # 不采用leaky的电位变化模式
            delta_u_t[t] = syn_grad
            u = u_last + delta_u_t[t]
            delta_u[t] = delta_u_t[t]
        else:
            u = (syn_m - syn_s) * theta_s / (theta_s - theta_m)
            delta_u[t] = u - u_last
            # 关注一下是否在梯度上发生改变
            delta_u_t[t] = syn_grad if is_grad_exp else delta_u[t]

        out = (u >= threshold).to(u)
        # 放电，重置
        u_last = u * (1 - out)

        #连带全部重置
        syn_m = syn_m * (1 - out)
        syn_s = syn_s * (1 - out)
        syn_grad = syn_grad * (1 - out)
        # 这里本质还是等间距离散时间步的脉冲发放，论文里的讨论基于理想的连续情况
        outputs[t] = out

    return delta_u, delta_u_t, outputs


@torch.jit.script
def neuron_backward_py(grad_delta, outputs, delta_u, delta_u_t, syn_a, partial_a, max_dudt_inv):
    # syn_a是global_v里预先计算好的常量，这里当做参数输入了
    T = grad_delta.shape[0]

    grad_in_, grad_w_ = torch.zeros_like(outputs), torch.zeros_like(outputs)
    # u对w和t的导数
    partial_u_grad_w, partial_u_grad_t = torch.zeros_like(outputs[0]), torch.zeros_like(outputs[0])
    delta_t = torch.zeros(outputs.shape[1:], device=outputs.device, dtype=torch.long)
    spiked = torch.zeros_like(outputs[0])

    for t in range(T - 1, -1, -1):
        # 反着来算梯度
        out = outputs[t]
        # 是否发放过脉冲
        spiked += (1 - spiked) * out

        # dt/du =(approximation) -du/dt
        partial_u = torch.clamp(-1 / delta_u[t], -4, 0)
        partial_u_t = torch.clamp(-1 / delta_u_t[t], -max_dudt_inv, 0)
        # current time is t_m
        # 如果此时发放了(这里是out的发放问题)，本次的梯度与后续的脉冲自然而然是毫无关系的，所以就置0，然后补上本次应有的计算结果
        partial_u_grad_w = partial_u_grad_w * (1 - out) + grad_delta[t] * partial_u * out
        partial_u_grad_t = partial_u_grad_t * (1 - out) + grad_delta[t] * partial_u_t * out

        # 一旦发放，由于只关注相对关系，就将delta_t调整到新的“相对距离”上
        delta_t = (delta_t + 1) * (1 - out).long()
        # spiked.to(partial_a), 将spiked转化为partial_a对应的数据类型和所处设备
        grad_in_[t] = partial_u_grad_t * partial_a[delta_t] * spiked.to(partial_a)
        grad_w_[t] = partial_u_grad_w * syn_a[delta_t] * spiked.to(syn_a)

    return grad_in_, grad_w_


def neuron_forward(in_I, neuron_config):
    theta_m, theta_s, theta_grad, threshold = torch.tensor(neuron_config).to(in_I)
    assert (theta_m != theta_s)
    is_grad_exp = torch.tensor(glv.network_config['gradient_type'] == 'exponential')
    is_forward_leaky = torch.tensor(glv.network_config['forward_type'] == 'leaky')
    if glv.network_config['backend'] == 'python':
        return neuron_forward_py(in_I, theta_m, theta_s, theta_grad, threshold, is_forward_leaky, is_grad_exp)
    elif glv.network_config['backend'] == 'cuda':
        # global neuron_cuda
        # if neuron_cuda is None:
        theta_m, theta_s, theta_grad, threshold = neuron_config
        return neuron_cuda.forward(in_I, theta_m, theta_s, theta_grad, threshold, is_forward_leaky, is_grad_exp)
    else:
        raise Exception('Unrecognized computation backend.')


def neuron_backward(grad_delta, outputs, delta_u, delta_u_t):
    syn_a, partial_a = glv.syn_a.to(outputs), -glv.delta_syn_a.to(outputs)
    max_dudt_inv = torch.tensor(glv.network_config['max_dudt_inv'])
    if glv.network_config['backend'] == 'python':
        return neuron_backward_py(grad_delta, outputs, delta_u, delta_u_t, syn_a, partial_a, max_dudt_inv)
    elif glv.network_config['backend'] == 'cuda':
        max_dudt_inv = max_dudt_inv.item()
        return neuron_cuda.backward(grad_delta, outputs, delta_u, delta_u_t, syn_a, partial_a, max_dudt_inv)
    else:
        raise Exception('Unrecognized computation backend.')


if __name__ == '__main__':
    T = 12
    glv.rank = 0
    config = dict()
    config['gradient_type'] = 'exponential'
    config['forward_type'] = 'nonleaky'
    for key, val in zip(('n_steps', 'tau_s', 'tau_m', 'tau_grad', 'threshold'), (T, 7, 4, 3.5, 1)):
        config[key] = val
    glv.init(config, config)
    neuron_cuda = load(name="neuron_cuda", sources=["neuron_cuda.cpp", 'neuron_cuda_kernel.cu'], verbose=True)
    shape = (T, 50, 3, 32, 32)

    neuron_config = [1 / glv.network_config[key] for key in ('tau_m', 'tau_s', 'tau_grad')] + [
        glv.network_config['threshold']]
    in_I = torch.rand(*shape, device=torch.device('cuda'))
    glv.network_config['backend'] = 'python'
    delta_u_py, delta_u_t_py, outputs_py = neuron_forward(in_I, neuron_config)
    glv.network_config['backend'] = 'cuda'
    delta_u_cuda, delta_u_t_cuda, outputs_cuda = neuron_forward(in_I, neuron_config)
    print(torch.sum(delta_u_py), torch.sum(delta_u_cuda))
    assert (torch.sum(torch.abs(delta_u_py - delta_u_cuda)).item() <= 1e-3)
    assert (torch.sum(torch.abs(delta_u_t_py - delta_u_t_cuda)).item() <= 1e-3)
    assert (torch.sum(torch.abs(outputs_py - outputs_cuda)) <= 1e-3)

    grad_delta = torch.rand(*shape, device=torch.device('cuda'))
    outputs = torch.round(torch.rand_like(grad_delta))
    delta_u = torch.rand_like(grad_delta) * 8 - 4
    delta_u_t = torch.rand_like(grad_delta) * 8 - 4
    glv.network_config['backend'] = 'python'
    grad_in_py, grad_w_py = neuron_backward(grad_delta, outputs, delta_u, delta_u_t)
    glv.network_config['backend'] = 'cuda'
    grad_in_cuda, grad_w_cuda = neuron_backward(grad_delta, outputs, delta_u, delta_u_t)
    print(torch.sum(grad_in_py), torch.sum(grad_in_cuda))
    assert (torch.sum(torch.abs(grad_in_py - grad_in_cuda)) <= 1e-3)
    assert (torch.sum(torch.abs(grad_w_py - grad_w_cuda)) <= 1e-3)
