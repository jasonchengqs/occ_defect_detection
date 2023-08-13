import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

import csv
import copy


CLASS_NAMES = ['MT_Blowhole', 'MT_Break', 'MT_Crack', 'MT_Free', 'MT_Uneven']


class SMDDataset(Dataset):
    def __init__(self, dataset_path='/mnt/ml/DATASET/SMD/', is_train=True,
                 resize=256, cropsize=256):
        # assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.dataset_path = dataset_path
        # self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize

        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder()

        # set transforms
        # self.transform_x = T.Compose([T.Resize(resize, Image.ANTIALIAS),
        #                               T.CenterCrop(cropsize),
        #                               T.ToTensor(),
        #                               T.Normalize(mean=[0.485, 0.456, 0.406],
        #                                           std=[0.229, 0.224, 0.225])])
        self.transform_x = T.Compose([
                                T.Resize(resize, Image.ANTIALIAS),
                                T.RandomCrop(cropsize),
                                T.ToTensor(),
                            ])
        self.transform_mask = T.Compose([T.Resize(resize, Image.NEAREST),
                                         T.CenterCrop(cropsize),
                                         T.ToTensor()])

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]

        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)
        y = torch.tensor(int(y))
        return x, y
        # if y == 0:
        #     mask = torch.zeros([1, self.cropsize, self.cropsize])
        # else:
        #     mask = Image.open(mask)
        #     mask = self.transform_mask(mask)
        # return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []
        if phase == 'train':
            select_cls = ['MT_Free']
        else:
            select_cls = ['MT_Blowhole', 'MT_Break', 'MT_Crack', 'MT_Uneven']
        for class_name in select_cls:
            img_dir = os.path.join(self.dataset_path, class_name, 'Imgs')
            gt_dir = os.path.join(self.dataset_path, class_name, 'Imgs')
    
 
            img_fpath_list = sorted([os.path.join(img_dir, f)
                                     for f in os.listdir(img_dir)
                                     if f.endswith('.jpg')])
            x.extend(img_fpath_list)

            gt_fpath_list = sorted([os.path.join(img_dir, f)
                                     for f in os.listdir(img_dir)
                                     if f.endswith('.png')])
            mask.extend(gt_fpath_list)
            if phase == 'train':
                y.extend([0] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
            assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask)


def load_train_dataset(data_path,
                       validation_ratio=0.1,
                       batch_size=32,
                       img_resize=256, 
                       img_cropsize=256):
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_dataset = SMDDataset(data_path, is_train=True, resize=img_resize, cropsize=img_cropsize)
    img_nums = len(train_dataset)
    valid_num = int(img_nums * validation_ratio)
    train_num = img_nums - valid_num
    train_data, val_data = torch.utils.data.random_split(train_dataset, [train_num, valid_num])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, val_loader


def load_test_dataset(data_path,
                      batch_size=32,
                      img_resize=256, 
                      img_cropsize=256):
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    test_dataset = SMDDataset(data_path, is_train=False, resize=img_resize, cropsize=img_cropsize)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    
    return test_loader