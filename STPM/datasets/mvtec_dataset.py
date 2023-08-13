import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

import csv
import copy

# URL = 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz'
CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']


class MVTecDataset(Dataset):
    def __init__(self, dataset_path='D:/dataset/mvtec_anomaly_detection', class_name='bottle', is_train=True,
                 resize=256, cropsize=256):
        assert class_name in CLASS_NAMES or class_name == 'all', 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize

        # load dataset
        if self.class_name != 'all':
            self.x, self.y, self.mask = self.load_dataset_folder()
        else:
            self.load_all_folders()

        # set transforms
        self.transform_x = T.Compose([T.Resize(resize, Image.ANTIALIAS),
                                      T.CenterCrop(cropsize),
                                      T.ToTensor(),
                                      T.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])])
        self.transform_mask = T.Compose([T.Resize(resize, Image.NEAREST),
                                         T.CenterCrop(cropsize),
                                         T.ToTensor()])

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]

        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)

        if y == 0:
            mask = torch.zeros([1, self.cropsize, self.cropsize])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)

        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_all_folders(self):
        self.x, self.y, self.mask = [], [], []
        for class_name in CLASS_NAMES:
            self.class_name = class_name
            x, y, m = self.load_dataset_folder()
            self.x.extend(x)
            self.y.extend(y)
            self.mask.extend(m)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.png')])
            x.extend(img_fpath_list)

            # load gt labels
            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                 for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask)


def load_train_dataset(data_path, 
                       class_name, 
                       validation_ratio=0.1,
                       batch_size=32,
                       img_resize=256, 
                       img_cropsize=256):
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_dataset = MVTecDataset(data_path, class_name, is_train=True, resize=img_resize, cropsize=img_cropsize)
    img_nums = len(train_dataset)
    valid_num = int(img_nums * validation_ratio)
    train_num = img_nums - valid_num
    train_data, val_data = torch.utils.data.random_split(train_dataset, [train_num, valid_num])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, val_loader


def load_test_dataset(data_path, 
                      class_name,
                      batch_size=32,
                      img_resize=256, 
                      img_cropsize=256):
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    test_dataset = MVTecDataset(data_path, class_name=class_name, is_train=False, resize=img_resize, cropsize=img_cropsize)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    
    return test_loader