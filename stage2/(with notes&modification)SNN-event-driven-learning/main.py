import os
import sys
import shutil

import torch
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from network_parser import parse
from datasets import loadMNIST, loadCIFAR10, loadCIFAR100, loadFashionMNIST, loadSpiking, loadSMNIST
from datasets.utils import TTFS
import cnns
from utils import learningStats
import layers.losses as losses
import numpy as np
from datetime import datetime
from torch.nn.utils import clip_grad_norm_
import global_v as glv

from sklearn.metrics import confusion_matrix
import argparse

log_interval = 100
multigpu = False


def get_loss(network_config, err, outputs, labels):
    if network_config['loss'] in ['kernel', 'timing']:
        targets = torch.zeros_like(outputs)
        device = torch.device(glv.rank)
        # 在这两种误差类型下，desired_spikes需要手动指定，而不参考config了
        if T >= 8:
            [0,1,0,1,0,1,0,1,0,1]
            desired_spikes = torch.tensor([0, 1], device=device).repeat(T // 2)
            if T % 2 == 1:
                #[1,0,1,....0,1,0,1,....]
                desired_spikes = torch.cat([torch.zeros(1, device=device), desired_spikes])
        else:
            # 没那么多时间步，就希望全都发放脉冲了，除了最开头
            desired_spikes = torch.ones(T, device=device)
            desired_spikes[0] = 0
        for i in range(len(labels)):
            # 为每个样本的正确标签位置设置为期望的脉冲序列
            targets[..., i, labels[i]] = desired_spikes

    if network_config['loss'] == "count":
        # set target signal
        # 采用论文里面的count loss，就手动指定这些参数了
        desired_count = network_config['desired_count']
        undesired_count = network_config['undesired_count']
        targets = torch.ones_like(outputs[0]) * undesired_count
        for i in range(len(labels)):
            targets[i, labels[i]] = desired_count
        loss = err.spike_count(outputs, targets)
    elif network_config['loss'] == "kernel":
        loss = err.spike_kernel(outputs, targets)
    elif network_config['loss'] == "TET":
        # set target signal
        loss = err.spike_TET(outputs, labels)
    else:
        raise Exception('Unrecognized loss function.')
    # 最终返回一个loss值，而不是loss对象
    return loss.to(glv.rank)


def readout(output, T):
    # 确保早期脉冲的影响略大于晚期脉冲
    # 将每个样本在所有时间步的脉冲（经加权后）累加，得到每个样本在每个类别上的总加权脉冲计数
    output *= 1.1 - torch.arange(T, device=torch.device(glv.rank)).reshape(T, 1, 1) / T / 10
    return torch.sum(output, dim=0).detach()


def preprocess(inputs, network_config):
    inputs = inputs.to(glv.rank)
    if network_config['encoding'] == 'TTFS':
        # 把输入转变为TTFS的形式
        inputs = torch.stack([TTFS(data, T) for data in inputs], dim=0)
    if len(inputs.shape) < 5:
        # 看来形状至多不超过5个参数
        inputs = inputs.unsqueeze_(0).repeat(T, 1, 1, 1, 1)
    else:
        inputs = inputs.permute(1, 0, 2, 3, 4)
    return inputs


def train(network, trainloader, opti, epoch, states, err):
    train_loss, correct, total = 0, 0, 0
    cnt_oneof, cnt_unique = 0, 0
    network_config = glv.network_config
    batch_size = network_config['batch_size']
    # 混合精度训练的梯度缩放器
    scaler = GradScaler()
    # 记录训练开始时间，或许是timeElasped的来源
    start_time = datetime.now()

    forward_time, backward_time, data_time, other_time, glv.time_use = 0, 0, 0, 0, 0
    t0 = datetime.now() # 记录初始时刻
    num_batch = len(trainloader)
    batch_idx = 0
    for inputs, labels in trainloader:
        torch.cuda.synchronize()
        data_time += (datetime.now() - t0).total_seconds()
        t0 = datetime.now() # 重新记录该epoch的起始时刻
        batch_idx += 1

        labels, inputs = (x.to(glv.rank) for x in (labels, inputs))
        inputs = preprocess(inputs, network_config)
        # forward pass
        if network_config['amp']:
            # amp 混合精度训练
            with autocast():
                # 网络计算
                outputs = network(inputs, labels, epoch, True)
                # 计算loss
                loss = get_loss(network_config, err, outputs, labels)
        else:
            outputs = network(inputs, labels, epoch, True)
            loss = get_loss(network_config, err, outputs, labels)
        assert (len(outputs.shape) == 3)

        torch.cuda.synchronize()
        # 前向推理用时
        forward_time += (datetime.now() - t0).total_seconds()
        t0 = datetime.now()
        # backward pass
        # 反向传播
        opti.zero_grad() # 清零梯度,opti即optimizer，内有lr等设置
        if network_config['amp']:
            # amp 混合精度训练
            scaler.scale(loss).backward() # 缩放损失并反向传播
            scaler.unscale_(opti) # 还原梯度用于裁剪
            clip_grad_norm_(network.parameters(), 1) # 梯度裁剪（防止爆炸）
            scaler.step(opti) # 优化器更新参数
            scaler.update() # 更新缩放器状态
        else:
            loss.backward()
            clip_grad_norm_(network.parameters(), 1)
            opti.step()
        # (network.module if multigpu else network).weight_clipper()
        torch.cuda.synchronize()
        # 计算整个反向传播用时
        backward_time += (datetime.now() - t0).total_seconds()
        t0 = datetime.now()

        spike_counts = readout(glv.outputs_raw, T) # 脉冲求和
        predicted = torch.argmax(spike_counts, axis=1) # 取最大的为预测结果
        # 累计损失、样本数和准确数
        train_loss += torch.sum(loss).item()
        total += len(labels)
        correct += (predicted == labels).sum().item()

        # learningStats的training部分更新
        # 所以train函数其实只负责了一个epoch的训练
        states.training.correctSamples = correct
        states.training.numSamples = total
        states.training.lossSum += loss.to('cpu').data.item()

        # 细粒度准确率分析（部分正确/完全正确）
        labels = labels.reshape(-1)
        idx = torch.arange(labels.shape[0], device=torch.device(glv.rank))
        # 提取出来正确类别的脉冲数
        nspike_label = spike_counts[idx, labels]
        # 部分正确：正确类别脉冲数等于最大脉冲数（可能有并列）
        cnt_oneof += torch.sum(nspike_label == torch.max(spike_counts, axis=1).values).item()
        # 临时降低正确类别的计数
        spike_counts[idx, labels] -= 1
        # 若nspike_label > torch.max(spike_counts, axis=1).values，则说明是独一的最大值，判断的十分准确，并无和其他类并列之嫌
        cnt_unique += torch.sum(nspike_label > torch.max(spike_counts, axis=1).values).item()
        spike_counts[idx, labels] += 1

        if (not multigpu or dist.get_rank() == 0) and (batch_idx % log_interval == 0 or batch_idx == num_batch):
            # if batch_idx % log_interval == 0:
            # 目前在第几个epoch，第几个batch，以及用时
            states.print(epoch, batch_idx, (datetime.now() - start_time).total_seconds())
            print('Time consumed on loading data = %.2f, forward = %.2f, backward = %.2f, other = %.2f'
                  % (data_time, forward_time, backward_time, other_time))
            data_time, forward_time, backward_time, other_time, glv.time_use = 0, 0, 0, 0, 0

            # 在这一个batch里面，部分正确和绝对正确的比例
            avg_oneof, avg_unique = cnt_oneof / (batch_size * batch_idx), cnt_unique / (batch_size * batch_idx)
            print(
                'Percentage of partially right = %.2f%%, entirely right = %.2f%%' % (avg_oneof * 100, avg_unique * 100))
            print()
        torch.cuda.synchronize()
        other_time += (datetime.now() - t0).total_seconds()
        t0 = datetime.now()

    acc = correct / total
    train_loss = train_loss / total

    return acc, train_loss


def test(network, testloader, epoch, states, log_dir):
    global best_acc # 全局变量，记录历史最佳测试准确率
    correct = 0
    total = 0
    network_config = glv.network_config
    T = network_config['n_steps']
    n_class = network_config['n_class']
    time = datetime.now()
    y_pred = []
    y_true = []
    num_batch = len(testloader)
    batch_idx = 0
    for inputs, labels in testloader:
        batch_idx += 1
        inputs = preprocess(inputs, network_config)
        # forward pass
        labels = labels.to(glv.rank)
        inputs = inputs.to(glv.rank)
        with torch.no_grad():
            outputs = network(inputs, None, epoch, False)

        spike_counts = readout(outputs, T).cpu().numpy()
        predicted = np.argmax(spike_counts, axis=1)
        labels = labels.cpu().numpy()
        y_pred.append(predicted)
        y_true.append(labels)
        total += len(labels)
        correct += (predicted == labels).sum().item()
        # learningStats的testing部分更新
        states.testing.correctSamples += (predicted == labels).sum().item()
        states.testing.numSamples = total
        if batch_idx % log_interval == 0 or batch_idx == num_batch:
            states.print(epoch, batch_idx, (datetime.now() - time).total_seconds())
    print()
    # 拼接所有批次的预测结果和真实标签
    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    # 计算混淆矩阵（归一化到每个类别的真实样本数）
    nums = np.bincount(y_true) # 统计每个类别的真实样本数
    # 计算混淆矩阵，分析各个类别的判断精度
    confusion = confusion_matrix(y_true, y_pred, labels=np.arange(n_class)) / nums.reshape(-1, 1)

    test_acc = correct / total

    state = {
        'net': (network.module if multigpu else network).state_dict(),
        'epoch': epoch,
    }
    torch.save(state, os.path.join(log_dir, 'last.pth'))

    if test_acc > best_acc:
        best_acc = test_acc
        # 保留测试集上表现最好的模型
        torch.save(state, os.path.join(log_dir, 'best.pth'))
    return test_acc, confusion


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # config是yaml文件的路径
    parser.add_argument('-config', action='store', dest='config', help='The path of config file')
    parser.add_argument('-checkpoint', action='store', dest='checkpoint',
                        help='The path of checkpoint, if use checkpoint')
    parser.add_argument('-seed', type=int, default=3, help='random seed (default: 3)')
    parser.add_argument('-dist', type=str, default="nccl", help='distributed data parallel backend')
    parser.add_argument('--local_rank', type=int, default=-1)
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        exit(0)

    if args.config is None:
        raise Exception('Unrecognized config file.')
    else:
        config_path = args.config

    # network_parser.py中的parse类，读取相关文件
    params = parse(config_path)

    # check GPU
    if not torch.cuda.is_available():
        print('No GPU device available')
        sys.exit(1)
    # set GPU
    if args.local_rank >= 0:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend=args.dist)
        glv.rank = args.local_rank
        multigpu = True
    else:
        glv.rank = 0
    cudnn.benchmark = True
    cudnn.enabled = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # 全局变量初始化
    # network_config = params['Network'], layers_config = params['Layers']
    # 参见mnist.yaml当中的划分
    glv.init(params['Network'], params['Layers'] if 'Layers' in params.parameters else None)
    
    # datasets文件夹下的一众读取相关数据集的函数
    data_path = os.path.expanduser(params['Network']['data_path'])
    dataset_func = {"MNIST": loadMNIST.get_mnist,
                    "NMNIST": loadSpiking.get_nmnist,
                    "FashionMNIST": loadFashionMNIST.get_fashionmnist,
                    "CIFAR10": loadCIFAR10.get_cifar10,
                    "CIFAR100": loadCIFAR100.get_cifar100,
                    "DVS128Gesture": loadSpiking.get_dvs128_gesture,
                    "CIFAR10DVS": loadSpiking.get_cifar10_dvs,
                    "SMNIST": loadSMNIST.get_smnist}
    try:
        # 读取训练集和测试集
        trainset, testset = dataset_func[params['Network']['dataset']](data_path, params['Network'])
    except:
        raise Exception('Unrecognized dataset name.')
    batch_size = params['Network']['batch_size']
    if multigpu:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=4,
                                                   sampler=train_sampler, pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4,
                                                   pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4,
                                              pin_memory=True)
    # 在RESNET.yaml和TTFS.yaml中涉及model_import
    if 'model_import' not in glv.network_config:
        # 不导入模型，就按照network_config来逐层设定
        net = cnns.Network(list(train_loader.dataset[0][0].shape[-3:])).to(glv.rank)
    else:
        exec(f"from {glv.network_config['model_import']} import Network")
        net = Network().to(glv.rank)
        print(net)

    T = params['Network']['t_train']
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cuda:0')
        net.load_state_dict(checkpoint['net'])
        epoch_start = checkpoint['epoch'] + 1
        print('Network loaded.')
        print(f'Start training from epoch {epoch_start}.')
    else:
        inputs = torch.stack([train_loader.dataset[i][0] for i in range(batch_size)], dim=0).to(glv.rank)
        # 调整输入的格式
        inputs = preprocess(inputs, glv.network_config)
        print("Start to initialize.")
        # initialize weights
        net.eval()
        # 初始化标符调整为true，表示正在初始化
        glv.init_flag = True
        # 进行初始化
        net(inputs, None, None, False)
        # 不知晓net.train()具体是否产生作用
        net.train()
        glv.init_flag = False
        epoch_start = 1

    # 设置loss对象
    error = losses.SpikeLoss().to(glv.rank)  # the loss is not defined here
    if multigpu:
        net = DDP(net, device_ids=[glv.rank], output_device=glv.rank)
    optim_type, weight_decay, lr = (glv.network_config[x] for x in ('optimizer', 'weight_decay', 'lr'))
    assert (optim_type in ['SGD', 'Adam', 'AdamW'])

    # norm_param, weight_param = net.get_parameters()
    optim_dict = {'SGD': torch.optim.SGD,
                  'Adam': torch.optim.Adam,
                  'AdamW': torch.optim.AdamW}
    norm_param, param = [], []
    for layer in net.modules():
        if layer.type in ['conv', 'linear']:
            norm_param.extend([layer.norm_weight, layer.norm_bias])
            param.append(layer.weight)
    # 设定优化器
    optimizer = optim_dict[optim_type]([
        {'params': param},
        {'params': norm_param, 'lr': lr * glv.network_config['norm_grad']}
        ], lr=lr, weight_decay=weight_decay)
    # 设定schedular
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=glv.network_config['epochs'])

    best_acc = 0

    l_states = learningStats()

    log_dir = f"{params['Network']['log_path']}_{datetime.now().strftime('%Y%m%d-%H%M%S')}/"
    writer = SummaryWriter(log_dir)
    confu_mats = []
    for path in ['logs']:
        if not os.path.isdir(path):
            os.mkdir(path)
    shutil.copyfile(config_path, os.path.join(log_dir, os.path.split(config_path)[-1]))

    (net.module if multigpu else net).train()
    # 每一个epoch，进行一次训练和测试
    for epoch in range(epoch_start, params['Network']['epochs'] + epoch_start):
        if multigpu:
            train_loader.sampler.set_epoch(epoch)
        # 一次训练
        l_states.training.reset()
        train_acc, loss = train(net, train_loader, optimizer, epoch, l_states, error)
        l_states.training.update()
        # 一次测试
        l_states.testing.reset()
        test_acc, confu_mat = test(net, test_loader, epoch, l_states, log_dir)
        l_states.testing.update()
        # 调整学习率
        lr_scheduler.step()

        # l_states的plot和save等功能确实没有被使用
        
        confu_mats.append(confu_mat)
        if glv.rank == 0:
            # 在tensorboard中记录（本次epoch最终返回的）训练和测试的准确率以及loss
            writer.add_scalars('Accuracy', {'train': train_acc,
                                            'test': test_acc}, epoch)
            writer.add_scalars('Loss', {'loss': loss}, epoch)
        np.save(log_dir + 'confusion_matrix.npy', np.stack(confu_mats))
