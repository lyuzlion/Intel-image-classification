import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


def get_data(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((args.image_size, args.image_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomErasing(scale=(0.04, 0.2), ratio=(0.5, 2)),
        torchvision.transforms.RandomCrop(32, padding=4),
    ])
    train_set = torchvision.datasets.ImageFolder(args.train_dataset_path, transform=transforms)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((args.image_size, args.image_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    valid_set = torchvision.datasets.ImageFolder(args.valid_dataset_path, transform=transforms)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True)
    return train_loader, valid_loader

