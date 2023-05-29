"""
作者: ksy
日期: 2023年 05月 28日
"""
"""
作者: ksy
日期: 2023年 05月 27日
"""
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import Resnet

res18 = Resnet.ResNet18(Resnet.BasicBlock)

transform = transforms.Compose(
    [transforms.ToTensor()])

batch_size = 250
train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                         download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=False, transform=transform)
testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

if __name__ == '__main__':

    images, labels = next(iter(trainloader))
    images = images[:3]

    writer = SummaryWriter("./pics3")

    alpha=1
    images = images.cuda()  # 数据转移到 GPU 上
    labels = labels.cuda()  # 标签转移到 GPU 上
    images, labels_a, labels_b, lam = mixup_data(images, labels, alpha, True)

    writer.add_images("images", images, global_step=0)

    writer.close()
