# HW2
使用CNN网络模型(自己设计或使用现有的CNN架构，如AlexNet，ResNet-18)作为baseline在CIFAR-100上训练并测试；对比cutmix, cutout, mixup三种方法以及baseline方法在CIFAR-100图像分类任务中的性能表现；对三张训练样本分别经过cutmix, cutout, mixup后进行可视化，一共show 9张图像。

CIFAR-10是一个更接近普适物体的彩色图像数据集。一共十个类别，每个类别有6000个图像，数据集中一共有50000 张训练图片和10000 张测试图片。

Baseline:
Batch_size选择了250，learning_rate选择了0.001，优化器选择了Adam优化器，epoch选择了20，loss_function选择了交叉熵损失函数。

Mixup:将随机的两张样本按比例混合，分类的结果按比例分配；
Cutout:随机的将样本中的部分区域cut掉，并且填充0像素值，分类的结果不变；
CutMix:就是将一部分区域cut掉但不填充0像素而是随机填充训练集中的其他数据的区域像素值，分类结果按一定的比例分配
