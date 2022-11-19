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


valset = CIFAR100(root="./data", train=False,
                  transform=test_transform, download=True)

batch_size = 256
num_workers = 16
val_sampler = None
val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                         shuffle=(val_sampler is None),
                                         sampler=val_sampler,
                                         num_workers=num_workers,
                                         persistent_workers=True)
