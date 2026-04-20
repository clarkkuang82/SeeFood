import random

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import Food101

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

NUM_CLASSES = 101
VAL_FRACTION = 0.10
SPLIT_SEED = 42


def build_train_transform(image_size=224):
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_eval_transform(image_size=224, resize=256):
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_food101_splits(root, image_size=224, download=True):
    train_full = Food101(root=root, split="train",
                         transform=build_train_transform(image_size), download=download)
    test = Food101(root=root, split="test",
                   transform=build_eval_transform(image_size), download=download)

    n_val = int(len(train_full) * VAL_FRACTION)
    n_train = len(train_full) - n_val
    gen = torch.Generator().manual_seed(SPLIT_SEED)
    train, val = random_split(train_full, [n_train, n_val], generator=gen)
    return train, val, test


def build_loaders(train, val, test, batch_size=64, num_workers=4, pin_memory=True):
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader


# do not change seed for reproducibility
def set_seed(seed=SPLIT_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device
