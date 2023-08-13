#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 20:14:43 2018

@author: seukgyo
"""

import os
import numpy as np
import time
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.datasets import MNIST
from torch import optim
from torch.utils.data import DataLoader

import cae

from itertools import zip_longest
import csv
import matplotlib.pyplot as plt

from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score

import torchvision.transforms as transforms
from emdc.folderFilter import ImageFolderFilter, ImageFolderPickle
from tqdm import tqdm
import argparse

#%%
"""
DataLoader
"""
data_fold = 'data'

# if not os.path.isdir(data_fold):
#     os.makedirs(data_fold)
    
# train_set = MNIST(root=data_fold, train=True, download=True)
# test_set = MNIST(root=data_fold, train=False, download=True)

# train_data = train_set.train_data.numpy()
# train_label = train_set.train_labels.numpy()

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
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to models (default: none)')              
parser.add_argument('--save_path', default='', type=str, metavar='PATH',
                    help='path to save checkpoint (default: none)')

args = parser.parse_args()
args.model_path = '/home/qisen/Experiments/sdc_exp/emdc_exp/ocnn_exp/model/CAE.pth'


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # normal class - 4
# class4 = train_data[np.where(train_label==4), :, :]
# class4 = class4.transpose(1, 0, 2, 3)

# rand_idx = np.random.choice(len(class4), 220)
# class4 = class4[rand_idx, :, :, :]

# # anomaly class - 0, 7, 9
# class0 = train_data[np.where(train_label==0), :, :]
# class0 = class0.transpose(1, 0, 2, 3)

# rand_idx = np.random.choice(len(class0), 5)
# class0 = class0[rand_idx, :, :, :]

# class7 = train_data[np.where(train_label==7), :, :]
# class7 = class7.transpose(1, 0, 2, 3)

# rand_idx = np.random.choice(len(class7), 3)
# class7 = class7[rand_idx, :, :, :]

# class9 = train_data[np.where(train_label==9), :, :]
# class9 = class9.transpose(1, 0, 2, 3)

# rand_idx = np.random.choice(len(class9), 3)
# class9 = class9[rand_idx, :, :, :]



# normal_class = class4
# anomaly_class = np.concatenate((class0, class7, class9), axis=0)

"""
pretrained model
"""
# pretrained_model_path = 'model/CAE.pth'
pretrained_model_path = args.model_path

print('loading network...')
model = cae.CAE()
model.load_state_dict(torch.load(pretrained_model_path))
model = model.to(device)
#%%

model.eval()

encoder = model.encoder
for param in model.parameters():
    model.requires_grad = False
# """
# forward encoder
# """
# # normal encode
# normal_encode = []

# for normal_img in normal_class:
#     normal_img = np.reshape(normal_img, (1, 1, 28, 28))
#     normal_img = torch.FloatTensor(normal_img/255.)
#     normal_img = normal_img.to(device)
#     output = encoder(normal_img)
    
#     output = output.cpu()
#     output = output.detach().numpy()
#     normal_encode.append(output)
    
# normal_encode = np.array(normal_encode)
# normal_encode = np.reshape(normal_encode, (normal_encode.shape[0], normal_encode.shape[2]))

# # anomaly encode
# anomaly_encode = []

# for anomaly_img in anomaly_class:
#     anomaly_img = np.reshape(anomaly_img, (1, 1, 28, 28))
#     anomaly_img = torch.FloatTensor(anomaly_img/255.)
#     anomaly_img = anomaly_img.to(device)
#     output = encoder(anomaly_img)
    
#     output = output.cpu()
#     output = output.detach().numpy()
#     anomaly_encode.append(output)
    
# anomaly_encode = np.array(anomaly_encode)
# anomaly_encode = np.reshape(anomaly_encode, (anomaly_encode.shape[0], anomaly_encode.shape[2]))

train_transforms = transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
test_transforms = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    ])
target_transforms = transforms.Compose([transforms.ToTensor()])
train_datasets = ImageFolderPickle(
    args.data_path, True, 
    transform=train_transforms,
    sample_ratio=0.5)
train_loader = DataLoader(train_datasets, 
    batch_size=args.batch_size, shuffle=True, num_workers=4)

test_datasets = ImageFolderPickle(
    args.data_path, False, 
    transform=test_transforms)
val_loader = DataLoader(test_datasets, 
    batch_size=args.batch_size, shuffle=False, num_workers=4) 

#%%
"""
train oc-nn
"""
"""
oc-nn model
"""
oc_nn_model_path = 'model/oc_nn.pth'
oc_nn_model_path = os.path.join(args.save_path, oc_nn_model_path)

x_size = 32
h_size = 32
y_size = 1

class OC_NN(nn.Module):
    def __init__(self):
        super(OC_NN, self).__init__()
        
        self.dense_out1 = nn.Linear(x_size, h_size)
        self.out2 = nn.Linear(h_size, y_size)
        
    def forward(self, img):
        w1 = self.dense_out1(img)
        w2 = self.out2(w1)
        
        return w1, w2

model = OC_NN()
model.to(device)

theta = np.random.normal(0, 1, h_size + h_size * x_size + 1)
rvalue = np.random.normal(0, 1, (len(train_loader), y_size))
nu = 0.04

def nnscore(x, w, v):
    # print(x.shape, w.shape, v.shape)
    return torch.matmul(torch.matmul(x, w.T), v)

def ocnn_loss(theta, x, nu, w1, w2, r):
    term1 = 0.5 * torch.sum(w1**2)
    term2 = 0.5 * torch.sum(w2**2)
    term3 = 1/nu * torch.mean(F.relu(r - nnscore(x, w1, w2)))
    term4 = -r
    
    return term1 + term2 + term3 + term4

# optimizer = optim.SGD(model.parameters(), lr=args.lr)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

dataset_size = len(train_loader)
# normal_encode = torch.FloatTensor(normal_encode/255.)

# train_loader = DataLoader(normal_encode, batch_size=32, shuffle=True, num_workers=4, drop_last=True)

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
    i = 0
    for _, inputs, _ in tqdm(train_loader):   
        inputs = inputs.to(device)
        inputs = encoder(inputs)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        w1, w2 = model(inputs)
        # print('w', w1, w2)
        r = nnscore(inputs, w1, w2)
        # print('r', r)        
        loss = ocnn_loss(theta, inputs, nu, w1, w2, r)
        # print('l', loss) 
        loss = loss.mean()
        # print('l', loss)
        # backward + optimize only if in training phase
        loss.backward()
        optimizer.step()
        # if i >= 4:
        #     raise ValueError('not true')
        # i += 1
        # statistics
        running_loss += loss.item() * inputs.size(0)

    r = r.cpu().detach().numpy()
    r = np.percentile(r, q=100*nu)
    epoch_loss = running_loss / dataset_size

    print('Loss: {:.4f} '.format(epoch_loss))
    print('Epoch = %d, r = %f'%(epoch+1, r))

    # deep copy the model
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_model_wts)
        torch.save(model.state_dict(), oc_nn_model_path)
        
    
    print()

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
print('Best Loss: {:4f}'.format(best_loss))






# # normal encode
# normal_encode = []

# for _, normal_img, _ in train_loader:
#     # normal_img = np.reshape(normal_img, (1, 1, 28, 28))
#     # normal_img = torch.FloatTensor(normal_img/255.)
#     normal_img = normal_img.to(device)
#     output = encoder(normal_img)
    
#     output = output.cpu()
#     output = output.detach().numpy()
#     normal_encode.append(output)
    
# normal_encode = np.array(normal_encode)
# normal_encode = np.reshape(normal_encode, (normal_encode.shape[0], normal_encode.shape[2]))

# anomaly encode
test_score = []
test_label = []
for _, img, lbs in tqdm(val_loader):
    img = img.to(device)
    output = encoder(img)
    # print('out:', output)
    # print('w1', w1)
    # print('w2', w2)
    score = nnscore(output, w1, w2)
    score = score.cpu().detach().numpy() - r
    
    lbs = lbs.cpu().detach().numpy()

    for sc in score:
        # print('sc:', sc)
        test_score.append(sc)
    for lb in lbs:
        # print('lb:', lb)
        test_label.append(lb)
    # output = output.cpu()

    # output = output.detach().numpy()
    # anomaly_encode.append(output)
    
# anomaly_encode = np.array(anomaly_encode)
# anomaly_encode = np.reshape(anomaly_encode, (anomaly_encode.shape[0], anomaly_encode.shape[2]))

# normal_encode = normal_encode.to(device)
# train_score = nnscore(normal_encode, w1, w2)
# train_score = train_score.cpu().detach().numpy() - r

# anomaly_encode = torch.FloatTensor(anomaly_encode)
# anomaly_encode = anomaly_encode.to(device)

# test_score = nnscore(anomaly_encode, w1, w2)
# test_score = test_score.cpu().detach().numpy() - r

#%%
"""
Write Decision Scores to CSV
"""
decision_score_path = 'res/oc-nn_linear.csv'
decision_score_path = os.path.join(args.save_path, decision_score_path)

print ('Writing file to ', decision_score_path)

# poslist = train_score.tolist()
# neglist = test_score.tolist()

# d = [poslist, neglist]
# export_data = zip_longest(*d, fillvalue='')
d = [test_score, test_label]
export_data = zip_longest(*d, fillvalue='')
with open(decision_score_path, 'w') as myfile:
    wr = csv.writer(myfile)
    wr.writerow(("score", "label"))
    wr.writerows(export_data)
myfile.close()

#%%
"""
Plot Decision Scores
"""
# plt.plot()
# plt.title("One Class NN", fontsize="x-large", fontweight='bold');
# plt.hist(train_score, bins=25, label='Normal')
# plt.hist(test_score, bins=25, label='Anomaly')

#%%
## Obtain the Metrics AUPRC, AUROC, P@10
# y_train = np.ones(train_score.shape[0])
# y_test = np.zeros(test_score.shape[0])
# y_true = np.concatenate((y_train, y_test))

# y_score = np.concatenate((train_score, test_score))

y_true = test_label
y_score = test_score

average_precision = average_precision_score(y_true, y_score)

print('Average precision-recall score: {0:0.4f}'.format(average_precision))

roc_score = roc_auc_score(y_true, y_score)

print('ROC score: {0:0.4f}'.format(roc_score))

# def compute_precAtK(y_true, y_score, K = 10):

#     if K is None:
#         K = y_true.shape[0]

#     # label top K largest predicted scores as + one's've

#     idx = np.argsort(y_score)
#     predLabel = np.zeros(y_true.shape)

#     predLabel[idx[:K]] = 1

#     prec = precision_score(y_true, predLabel)

#     return prec

# prec_atk = compute_precAtK(y_true, y_score)

# print('Precision AtK: {0:0.4f}'.format(prec_atk))
