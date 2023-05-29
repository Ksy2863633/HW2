"""
作者: ksy
日期: 2023年 05月 29日
"""
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
    [transforms.ToTensor()])

batch_size = 250
train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                         download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=False, transform=transform)
testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

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

    images, labels = next(iter(trainloader))
    images = images[:3]

    writer = SummaryWriter("./pics2")

    for i in range(images.size(0)):
        if random.random() < 0.5:
            images[i] = cut(images[i])
    images = images.cuda()  # 数据转移到 GPU 上
    labels = labels.cuda()  # 标签转移到 GPU 上

    writer.add_images("images", images, global_step=0)

    writer.close()
