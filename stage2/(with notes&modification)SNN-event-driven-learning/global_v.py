import torch

# 总的来说这就是一个配置许多全局变量的代码集中地

def init(config_n, config_l=None):
    # 先申请许多全局变量
    # 里面还有T tau_s tau_m等论文里出现的重要变量
    # 甚至还有syn_a这个膜电位相关的变量
    global T, T_train, syn_a, delta_syn_a, tau_s, tau_m, grad_rec, outputs_raw
    global rank, network_config, layers_config, time_use, req_grad, init_flag
    init_flag = False
    
    # 这两个是全局变量
    network_config, layers_config = config_n, config_l

    if 'loss_reverse' not in network_config.keys():
        network_config['loss_reverse'] = True

    if 'encoding' not in network_config.keys():
        network_config['encoding'] = 'None'
    if 'amp' not in network_config.keys():
        network_config['amp'] = False
    if 'backend' not in network_config.keys():
        network_config['backend'] = 'python'
    if 'norm_grad' not in network_config.keys():
        network_config['norm_grad'] = 1

    # max_dudt_inv 该变量在神经元反向传播计算中发挥作用，具体见于 layers/functions.py 的 neuron_backward_py
    # 它是用于保障反向传播数值稳定性的超参数，主要作用是限制梯度计算中某一中间变量的取值范围，避免数值溢出或异常
    if 'max_dudt_inv' not in network_config:
        network_config['max_dudt_inv'] = 123456789
    if 'avg_spike_init' not in network_config:
        network_config['avg_spike_init'] = 1
    if 'weight_decay' not in network_config:
        network_config['weight_decay'] = 0
    if 't_train' not in network_config:
        network_config['t_train'] = network_config['n_steps']

    # 一行做到四行的事
    T, tau_s, tau_m, grad_type = (config_n[x] for x in ('n_steps', 'tau_s', 'tau_m', 'gradient_type'))

    if 'forward_type' not in network_config:
        # 未指定，则leaky
        network_config['forward_type'] = 'leaky'

    # 防意外
    # 注意到这里的grad_type有两种可能性，一种是original，一种是exponential，也即论文提出的保正的指数式梯度
    assert(network_config['forward_type'] in ['leaky', 'nonleaky'])
    assert(grad_type in ['original', 'exponential'])
    # not 【又是nonleaky又是用original的梯度】(有一个计算符优先级的问题)
    assert(not (network_config['forward_type'] == 'nonleaky' and grad_type == 'original'))

    syn_a, delta_syn_a = (torch.zeros(T + 1, device=torch.device(rank)) for _ in range(2))
    theta_m, theta_s = 1 / tau_m, 1 / tau_s
    if grad_type == 'exponential':
        assert('tau_grad' in config_n)
        tau_grad = config_n['tau_grad']
        theta_grad = 1 / tau_grad

    for t in range(T):
        t1 = t + 1
        # syn_a是各个时间步时的衰减系数近似值
        # 其实就是论文中的kernel值的各种待选项，由于文中证明了相对关系是重要的，所以直接存下来一堆备选值待用了
        # 但是也看到我们好像只是在假装我们的非浮点运算很少
        # delta_syn_a是syn_a关于t的导数
        syn_a[t] = ((1 - theta_m) ** t1 - (1 - theta_s) ** t1) * theta_s / (theta_s - theta_m)
        if grad_type == 'exponential':
            delta_syn_a[t] = (1 - theta_grad) ** t1 # (h(t) = exp(-t/tau_grad) 的离散形式，是由一阶离散差分的方法得到的近似结果) 
        else:
            f = lambda t: ((1 - theta_m) ** t - (1 - theta_s) ** t) * theta_s / (theta_s - theta_m)
            delta_syn_a[t] = f(t1) - f(t1 - 1)
    # print(syn_a, delta_syn_a)