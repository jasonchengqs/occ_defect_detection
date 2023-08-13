import argparse
import sys
from tqdm import tqdm 

from model import ConvAutoencoder
import numpy as np

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from emdc.folderFilter import ImageFolderFilter, ImageFolderPickle
import os

# from model import build_cae_model


def parse_args():
    parser = argparse.ArgumentParser(description='Train Convolutional AutoEncoder and inference')
    parser.add_argument('--dataset_path', default='./data/cifar10.npz', type=str, help='path to dataset')
    parser.add_argument('--img_size', default=128, type=int, help='height of images')
    parser.add_argument('--crop_size', default=128, type=int, help='width of images')
    parser.add_argument('--channel', default=3, type=int, help='channel of images')
    parser.add_argument('--num_epoch', default=50, type=int, help='the number of epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='mini batch size')
    parser.add_argument('--save_path', default='./data/cifar10_cae.npz', type=str, help='path to directory to output')

    args = parser.parse_args()

    return args


def flat_feature(enc_out):
    """flat feature of CAE features
    """
    enc_out_flat = []

    s1, s2, s3 = enc_out[0].shape
    s = s1 * s2 * s3
    for con in enc_out:
        enc_out_flat.append(con.reshape((s,)))

    return np.array(enc_out_flat)

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

def main():
    """main function"""
    args = parse_args()
    dataset_path = args.dataset_path
    img_size = args.img_size
    crop_size = args.crop_size
    channel = args.channel
    num_epoch = args.num_epoch
    batch_size = args.batch_size
    save_path = args.save_path

    device = get_device()

    train_transforms = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(crop_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    test_transforms = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    train_datasets = ImageFolderPickle(
        args.dataset_path, True, 
        transform=train_transforms, sample_ratio=0.1)
    train_loader = DataLoader(train_datasets, 
        batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    test_datasets = ImageFolderPickle(
        args.dataset_path, False, 
        transform=test_transforms)
    test_loader = DataLoader(test_datasets, 
        batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # build model and train
    autoencoder = ConvAutoencoder()

    #Loss function
    criterion = nn.MSELoss(reduction='mean')

    #Optimizer
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.01)
    
    #Train
    autoencoder.train()
    autoencoder.to(device)

    for epoch in range(1, num_epoch+1):
        # monitor training loss
        train_loss = 0.0

        #Training
        for data in tqdm(train_loader):
            _, images, _ = data
            images = images.to(device)
            optimizer.zero_grad()
            _, outputs = autoencoder(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*images.size(0)
            
        train_loss = train_loss/len(train_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
    
    # inference from encoder
    autoencoder.eval()
    for param in autoencoder.parameters():
        param.requires_grad = False

    memo = None
    for data in train_loader:
        _, images, y = data
        images = images.to(device)
        enc_out, _ = autoencoder(images)

        # flat features for OC-SVM input
        # enc_out = flat_feature(enc_out)
        _, c, h, w = enc_out.shape
        feature_dim = c * h * w
        enc_out = enc_out.reshape(-1, feature_dim)
        if memo is None:
            memo = enc_out.cpu().numpy()
        else:
            memo = np.append(memo, enc_out.cpu().numpy(), axis=0)
        # print(memo.shape)
    # save cae output

    save_file = os.path.join(save_path, 'train_set.npz')
    np.savez(save_file, ae_out=memo, 
        labels=['1' for _ in range(memo.shape[0])])

    memo = None
    memo_label = None
    for data in test_loader:
        _, images, y = data
        images = images.to(device)
        enc_out, _ = autoencoder(images)

        # flat features for OC-SVM input
        # enc_out = flat_feature(enc_out)
        _, c, h, w = enc_out.shape
        feature_dim = c * h * w
        enc_out = enc_out.reshape(-1, feature_dim)
        if memo is None:
            memo = enc_out.cpu().numpy()
        else:
            memo = np.append(memo, enc_out.cpu().numpy(), axis=0)
        if memo_label is None:
            memo_label = np.array(y)
        else:
            memo_label = np.append(memo_label, np.array(y), axis=0)

    save_file = os.path.join(save_path, 'test_set.npz')
    np.savez(save_file, ae_out=memo, 
        labels=memo_label)

if __name__ == '__main__':
    main()
