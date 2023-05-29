"""
作者: ksy
日期: 2023年 05月 28日
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
import random

res18 = Resnet.ResNet18(Resnet.BasicBlock)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

batch_size = 250
train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                         download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=False, transform=transform)
testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
network = res18.cuda()

loss_func = nn.CrossEntropyLoss()  # 损失函数：交叉熵损失
optimizer = torch.optim.Adam(network.parameters(), lr=0.001)  # 优化器


def get_num_correct(preds, labels):  # 计算正确分类的次数
    return preds.argmax(dim=1).eq(labels).sum().item()


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    """1.论文里的公式2 求出B的rw,rh"""
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    """2.论文里的公式2 求出B的rx,ry(bbox的中心点)"""
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    # 限制坐标区域不超过样本大小

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    """3.返回剪裁B区域的坐标值"""
    return bbx1, bby1, bbx2, bby2


if __name__ == '__main__':

    writer1 = SummaryWriter("./runs2/loss")
    writer2 = SummaryWriter("./runs2/acc")

    total_epochs = 20  # 由于训练耗时较长，这次我们只训练 10 个周期来看一下结果

    for epoch in range(total_epochs):
        network = network.cuda()
        total_loss = 0
        total_train_correct = 0

        alpha = 1.0

        for batch in trainloader:  # 抓取一个 batch

            # 读取样本数据
            images, labels = batch
            images = images.cuda()  # 数据转移到 GPU 上
            labels = labels.cuda()  # 标签转移到 GPU 上

            if random.random() < 0.5:
                lam = np.random.beta(alpha, alpha)
                rand_index = torch.randperm(images.size()[0]).to(device)
                labels_a = labels
                labels_b = labels[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
                images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
            else:
                labels_a = labels
                labels_b = labels
                lam = 1.0

            # 完成正向传播，计算损失
            preds = network(images)
            loss = loss_func(preds, labels_a) * lam + loss_func(preds, labels_b) * (1. - lam)


            # 偏导归零
            optimizer.zero_grad()

            # 反向传播
            loss.backward()

            # 更新参数
            optimizer.step()

            total_loss += loss.item()
            total_train_correct += get_num_correct(preds, labels)

        writer1.add_scalar("loss", total_loss, epoch)
        writer2.add_scalar("acc_train", total_train_correct / len(train_set), epoch)

        print("epoch: ", epoch,
              "correct times:", total_train_correct,
              "training accuracy:", "%.3f" % (total_train_correct / len(train_set) * 100), "%",
              "total_loss:", "%.3f" % total_loss)

        network = network.cpu()
        num_correct = 0
        for i, batch in enumerate(testloader):
            images, labels = batch
            preds = network(images)
            num_correct += get_num_correct(preds, labels)
        test_accuracy = num_correct / 10000

        writer2.add_scalar("acc_test", test_accuracy, epoch)

        print("epoch: ", epoch,
              "correct times:", num_correct,
              "testing accuracy:", "%.3f" % (test_accuracy * 100), "%")

    torch.save(network.cpu(), "resnet18.pt")
    network = Resnet.ResNet18(Resnet.BasicBlock)
    network = torch.load("resnet18.pt")
    num_correct = 0
    for i, batch in enumerate(testloader):
        images, labels = batch
        preds = network(images)
        num_correct += get_num_correct(preds, labels)

    test_accuracy = num_correct / 10000
    print("final test accuracy: ", test_accuracy)

    writer1.close()
    writer2.close()
