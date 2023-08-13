import os
# import tarfile
from PIL import Image
# import urllib.request

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import csv
import copy
import pickle as pk
import numpy as np

# URL = 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz'
CLASS_NAMES = [
    'D6DRF', 'D6BU', 'D5OF', 'carpet', 'D6XC', 'D6XE1', 'leather', 'metal_nut', 'pill', 'screw', 'tile',
    'toothbrush', 'transistor', 'wood', 'zipper', 'D4NG'
]

CLASS_LABEL_MAP = {0:0,1:0,2:0,3:0,4:1}

class EMDCDataset(Dataset):
    def __init__(self,
                 dataset_path='',
                 phase='train',
                 sample_size=5000,
                 fold=0,
                 resize=256,
                 cropsize=256
                 ):
        self.dataset = pk.load(open(dataset_path, 'rb'))
        self.fold = fold
        self.phase = phase
        self.resize = resize
        self.sample_size = sample_size

        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder()

        # set transforms
        self.transform_x = transforms.Compose([
            transforms.Resize(resize, Image.ANTIALIAS),
            transforms.CenterCrop(cropsize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transform_mask = transforms.Compose([
            transforms.Resize(resize, Image.NEAREST),
            transforms.CenterCrop(cropsize),
            transforms.ToTensor()])

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]

        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)

        # if y == 0:
        #     mask = torch.zeros([1, self.resize, self.resize])
        # else:
        #     mask = Image.open(mask)
        #     mask = self.transform_mask(mask)

        return x, y

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        mask = []

        if self.phase == 'train':
            _data = self.dataset[self.fold]['train']
            y = [CLASS_LABEL_MAP[d[1]] for d in _data]
            idx = np.array(y) == 1
            y = np.array(y)[idx]
            x = np.array([d[0] for d in _data])[idx]
        elif self.phase == 'val':
            _data = self.dataset[self.fold]['test']
            y = [CLASS_LABEL_MAP[d[1]] for d in _data]
            idx = np.array(y) == 1
            y = np.array(y)[idx]
            x = np.array([d[0] for d in _data])[idx]
        elif self.phase == 'test':
            # _data = self.dataset[self.fold]['train']
            # y0 = [CLASS_LABEL_MAP[d[1]] for d in _data]
            # idx = np.array(y0) != 1
            # y0 = np.array(y0)[idx][:159]
            # x0 = np.array([d[0] for d in _data])[idx][:159]

            x = [d for d in self.dataset['test_samples']]
            y = [CLASS_LABEL_MAP[d] for d in self.dataset['test_labels']]
            idx = np.array(y) != 1
            x1 = np.array(x)[idx]
            y1 = np.array(y)[idx]

            x = np.concatenate([x1, np.array(x)[~idx]])
            y = np.concatenate([y1, np.array(y)[~idx]])

        assert len(x) == len(y), 'number of x and y should be same'

        return x[:self.sample_size], y[:self.sample_size], mask
