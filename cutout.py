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

class Cutout:
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for i in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

cut = Cutout(n_holes=1, length=16)

if __name__ == '__main__':

    writer1 = SummaryWriter("./runs3/loss")
    writer2 = SummaryWriter("./runs3/acc")

    total_epochs = 20  # 由于训练耗时较长，这次我们只训练 10 个周期来看一下结果

    for epoch in range(total_epochs):
        network = network.cuda()
        total_loss = 0
        total_train_correct = 0
        for batch in trainloader:  # 抓取一个 batch

            # 读取样本数据
            images, labels = batch
            for i in range(images.size(0)):
                if random.random() < 0.5:
                    images[i] = cut(images[i])
            images = images.cuda()  # 数据转移到 GPU 上
            labels = labels.cuda()  # 标签转移到 GPU 上

            # 完成正向传播，计算损失
            preds = network(images)
            loss = loss_func(preds, labels)

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
