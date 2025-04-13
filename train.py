import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils import *
from model import *
import torch
import torch.nn as nn
import numpy as np


parser = argparse.ArgumentParser()
args = parser.parse_args()


args.epochs = 300
args.batch_size = 256
args.image_size = 64
args.num_classes = 6
args.train_dataset_path = "/home/liuzilong/data/liuzilong/intel-image-classification/seg_train/seg_train/"
args.valid_dataset_path = "/home/liuzilong/data/liuzilong/intel-image-classification/seg_test/seg_test/"
args.device = "cuda"
args.lr = 0.005
args.save_path = "/home/liuzilong/data/liuzilong/checkpoints/intel-image-classification/"


device = args.device
train_loader, valid_loader = get_data(args)
model = ResNet50(num_classes=args.num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
loss_fn = nn.CrossEntropyLoss()
valid_loss_min = np.Inf

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

for epoch in tqdm(range(args.epochs)):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 1 == 0:
        model.eval()
        valid_loss = 0.0
        correct = 0
        test_num = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(valid_loader):
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                loss = loss_fn(logits, labels)
                valid_loss += loss.item() * images.size(0)
                predictions = torch.argmax(logits, 1)
                correct += sum(predictions == labels.to(device)).item()
                test_num += labels.size(0)
        print('now epoch :  ', epoch, '     |   accuracy :   ' , correct / test_num, flush=True)        

        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss), flush=True)
            torch.save(model.state_dict(), args.save_path + 'model.pt')
            valid_loss_min = valid_loss