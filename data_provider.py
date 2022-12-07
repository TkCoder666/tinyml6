import torch
import torchvision
import torchvision.transforms as T
from torchvision.datasets import CIFAR100
train_transform = T.Compose([
    T.ToTensor(),
])

test_transform = T.Compose([
    T.ToTensor(),
])


def get_train_transforms(mean, std):
    return T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])


def get_val_transforms(mean, std):
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])


def get_vit_train_transforms(mean, std, img_size):
    return T.Compose([
        T.RandomResizedCrop((img_size, img_size), scale=(0.05, 1.0)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def get_vit_val_transforms(mean, std, img_size):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


mean = [0.5070, 0.4865, 0.4409]
std = [0.2673, 0.2564, 0.2761]

valset = CIFAR100(root="./data", train=False,
                  transform=get_train_transforms(mean, std), download=True)
trainset = CIFAR100(root="./data", train=True,
                    transform=get_val_transforms(mean, std), download=True)


val_sampler = None


def get_test_loader(batch_size, num_workers):
    test_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                              shuffle=(val_sampler is None),
                                              sampler=val_sampler,
                                              num_workers=num_workers,
                                              persistent_workers=True)
    return test_loader


def get_train_loader(batch_size, num_workers):
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=(val_sampler is None),
                                               sampler=val_sampler,
                                               num_workers=num_workers,
                                               persistent_workers=True)
    return train_loader