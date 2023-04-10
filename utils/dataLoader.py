from torchvision.datasets import ImageFolder
from torchvision import transforms
from random import randint
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import cv2


class RotationDataLoader(Dataset):
    # 数据加载器
    def __init__(self, is_train, trans=None):
        if is_train:
            if trans is not None:
                dataset = datasets.CIFAR10(root='data/', train=True, transform=trans, download=True)
            else:
                dataset = datasets.CIFAR10(root='data/', train=True, download=True)
        else:
            if trans is not None:
                dataset = datasets.CIFAR10(root='data/', train=False, transform=trans, download=True)
            else:
                dataset = datasets.CIFAR10(root='data/', train=False, download=True)

        self.length = len(dataset)
        self.images = []
        self.labels = [i % 4 for i in range(self.length * 4)]
        for image, _ in dataset:
            img = image.permute(1, 2, 0).detach().numpy()
            img_90 = cv2.flip(cv2.transpose(img.copy()), 1)
            img_180 = cv2.flip(cv2.transpose(img_90.copy()), 1)
            img_270 = cv2.flip(cv2.transpose(img_180.copy()), 1)
            self.images += [torch.tensor(img).permute(2, 0, 1), torch.tensor(img_90).permute(2, 0, 1),
                            torch.tensor(img_180).permute(2, 0, 1), torch.tensor(img_270).permute(2, 0, 1)]

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.length


def LoadRotationDataset(batch_size, trans=None):
    if trans is not None:
        train_iter = DataLoader(RotationDataLoader(is_train=True, trans=trans), batch_size=batch_size, shuffle=True)
        test_iter = DataLoader(RotationDataLoader(is_train=False, trans=trans), batch_size=batch_size)
    else:
        train_iter = DataLoader(RotationDataLoader(is_train=True), batch_size=batch_size, shuffle=True)
        test_iter = DataLoader(RotationDataLoader(is_train=False), batch_size=batch_size)
    return train_iter, test_iter


def LoadSuperviseDataset(batch_size, trans=None):
    if trans is not None:
        train_dataset = datasets.CIFAR10(root='data/', train=True, transform=trans, download=True)
        test_dataset = datasets.CIFAR10(root='data/', train=False, transform=trans, download=True)
    else:
        train_dataset = datasets.CIFAR10(root='data/', train=True, download=True)
        test_dataset = datasets.CIFAR10(root='data/', train=False, download=True)

    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(test_dataset, batch_size=batch_size)
    return train_iter, test_iter
