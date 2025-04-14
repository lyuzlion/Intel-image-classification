import torch
import torch.nn as nn
import numpy as np
import torchvision
import argparse
from torch.utils.data import DataLoader
from model import ResNet50

parser = argparse.ArgumentParser()
args = parser.parse_args()

args.image_size = 64
args.test_dataset_path = "/home/liuzilong/data/liuzilong/intel-image-classification/seg_test/seg_test/"
args.batch_size = 256
args.device = "cuda"
args.num_classes = 6
args.model_path = "/home/liuzilong/data/liuzilong/checkpoints/intel-image-classification/model.pt"

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((args.image_size, args.image_size)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
test_set = torchvision.datasets.ImageFolder(args.test_dataset_path, transform=transforms)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

model = ResNet50(num_classes=args.num_classes).to(args.device)
model.load_state_dict(torch.load(args.model_path, weights_only=False))
model = model.to(args.device)

model.eval()
valid_loss = 0.0
correct = 0
test_num = 0
with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(args.device)
        labels = labels.to(args.device)
        logits = model(images)
        predictions = torch.argmax(logits, 1)
        correct += sum(predictions == labels.to(args.device)).item()
        test_num += labels.size(0)
print('accuracy :   ' , correct / test_num, flush=True)        
