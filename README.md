# NeuroAI-2025-Fall-Project

## Stage 1

## Stage 2
选择任务(1)，事件驱动的脉冲神经网络反向传播算法，参考论文[Training Spiking Neural Networks with Event-driven Backpropagation](https://proceedings.neurips.cc/paper_files/paper/2022/hash/c4e5f4de1b3cfc838eec6484d0b85378-Abstract-Conference.html)

包含的相关文件如下：
### (with notes&modification)SNN-event-driven-learning
项目原代码，附有阅读时的注释，以及Normalization on Weight相关错误的修正。
### curves of Loss&Accuracy
在mnist数据集上进行的复现实验的误差和准确度曲线的记录，单次实验共100个epoch，具体配置参考原代码中mnist.yaml文件，命名含fixed_bn前缀的是Normalization on Weight相关错误的修正后再次进行训练的相关记录。