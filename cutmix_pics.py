import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import random

batch_size=250
transform = transforms.Compose(
    [transforms.ToTensor()])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                         download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=False, transform=transform)
testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

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

    images, labels = next(iter(trainloader))
    images = images[:3]

    writer = SummaryWriter("./pics1")

    alpha = 1.0
    if random.random() < 0.5:
        lam = np.random.beta(alpha, alpha)
        rand_index = torch.randperm(images.size()[0])
        labels_a = labels
        labels_b = labels[rand_index]
        bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
        images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
    else:
        labels_a = labels
        labels_b = labels
        lam = 1.0

    writer.add_images("images", images, global_step=0)

    writer.close()
