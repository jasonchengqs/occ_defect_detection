#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 19:07:03 2018

@author: seukgyo
"""

import os
import torch
import numpy as np
import time
import copy
import argparse
import torch.nn.functional as F

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch import nn
from emdc.folderFilter import ImageFolderFilter, ImageFolderPickle
from tqdm import tqdm
from smd_dataset import load_train_dataset, load_test_dataset


"""
DataLoader
"""
data_fold = 'data'

# if not os.path.isdir(data_fold):
#     os.makedirs(data_fold)

parser = argparse.ArgumentParser(description='PyTorch Light CNN Training')
parser.add_argument('--cuda', '-c', default=True)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--epochs', default=80, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--data_path', default='', type=str, metavar='PATH',
                    help='path to root path of images (default: none)')
parser.add_argument('--save_path', default='', type=str, metavar='PATH',
                    help='path to save checkpoint (default: none)')

args = parser.parse_args()


train_loader, val_loader = load_train_dataset(
    args.data_path,
    validation_ratio=0.1,
    batch_size=args.batch_size,
    img_resize=128, 
    img_cropsize=128)

test_loader = load_test_dataset(
    args.data_path,
    batch_size=args.batch_size,
    img_resize=128, 
    img_cropsize=128)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset_size = len(train_loader)

#%%
"""
Model
"""
cae_model_path = 'model/CAE.pth'
cae_model_path = os.path.join(args.save_path, cae_model_path)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(8, 8, 3, stride=1, padding=1)
        
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        
        # self.dense1 = nn.Linear(392, 32)
        self.dense1 = nn.Linear(8192, 32)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(8)
        self.bn3 = nn.BatchNorm2d(8)
        
    def forward(self, img):
        x = self.conv1(img)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.elu(x)
        
        # x = x.view(-1, 392)
        x = x.view(-1, 8192)
        x = self.dense1(x)
        x = F.dropout(x, training=self.training)
        x = F.elu(x)
        
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.deconv3 = nn.ConvTranspose2d(8, 8, 3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(8, 16, 3, stride=1, padding=1)
        self.deconv1 = nn.ConvTranspose2d(16, 3, 3, stride=1, padding=1)
        
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')
        
        # self.dense1 = nn.Linear(32, 392)
        self.dense1 = nn.Linear(32, 8192)

        self.bn3 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        
    def forward(self, encode):
        x = self.dense1(encode)
        x = F.dropout(x, training=self.training)
        x = F.elu(x)
        
        # x = x.view(x.size(0), 8, 7, 7)
        x = x.view(x.size(0), 8, 32, 32)

        x = self.deconv3(x)
        x = self.bn3(x)
        x = F.elu(x)
        
        x = self.upsample2(x)
        
        x = self.deconv2(x)
        x = self.bn2(x)
        x = F.elu(x)
        
        x = self.upsample1(x)
        
        x = self.deconv1(x)
        x = F.sigmoid(x)
        
        return x
        
class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self, img):
        x = self.encoder(img)
        x = self.decoder(x)
        
        return x

"""
train
"""
if __name__ == '__main__':
    model = CAE()
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters())
    
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000
    
    num_epochs = args.epochs
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
    
        # Each epoch has a training and validation phase
        model.train()  # Set model to training mode
        
        running_loss = 0.0
    
        # Iterate over data.
        for inputs, _ in tqdm(train_loader):   
            inputs = inputs.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward
            # track history if only in train
            outputs = model(inputs)
            loss = F.mse_loss(inputs, outputs)
    
            # backward + optimize only if in training phase
            loss.backward()
            optimizer.step()
    
            # statistics
            running_loss += loss.item() * inputs.size(0)
    
        epoch_loss = running_loss / dataset_size
    
        print('Loss: {:.4f} '.format(epoch_loss))
    
        # deep copy the model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            model.load_state_dict(best_model_wts)
            torch.save(model.state_dict(), cae_model_path)
        print()
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best Loss: {:4f}'.format(best_loss))
