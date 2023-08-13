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
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper', 'D4NG']


class MVTecDataset(Dataset):
    def __init__(self, dataset_path='D:/dataset/mvtec_anomaly_detection', class_name='bottle', is_train=True,
                 resize=256, cropsize=256):
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize

        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder()

        # set transforms
        self.transform_x = T.Compose([T.Resize(resize, Image.ANTIALIAS),
                                      T.CenterCrop(cropsize),
                                      T.ToTensor(),
                                      T.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])])
        self.transform_mask = T.Compose([T.Resize(resize, Image.NEAREST),
                                         T.CenterCrop(cropsize),
                                         T.ToTensor()])

    # def __getitem__(self, idx):
    #     x, y, mask = self.x[idx], self.y[idx], self.mask[idx]

    #     x = Image.open(x).convert('RGB')
    #     x = self.transform_x(x)

    #     if y == 0:
    #         mask = torch.zeros([1, self.cropsize, self.cropsize])
    #     else:
    #         mask = Image.open(mask)
    #         mask = self.transform_mask(mask)

    #     return x, y, mask

    def __getitem__(self, idx):
        f, y, _ = self.x[idx], self.y[idx], self.mask[idx]

        x = Image.open(f).convert('RGB')
        x = self.transform_x(x)

        return x, y, f.split('/')[-1]
    
    def __len__(self):
        return len(self.x)

    # def load_dataset_folder(self):
    #     phase = 'train' if self.is_train else 'test'
    #     x, y, mask = [], [], []

    #     img_dir = os.path.join(self.dataset_path, self.class_name, phase)
    #     gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')

    #     img_types = sorted(os.listdir(img_dir))
    #     for img_type in img_types:

    #         # load images
    #         img_type_dir = os.path.join(img_dir, img_type)
    #         if not os.path.isdir(img_type_dir):
    #             continue
    #         img_fpath_list = sorted([os.path.join(img_type_dir, f)
    #                                  for f in os.listdir(img_type_dir)
    #                                  if f.endswith('.png')])
    #         x.extend(img_fpath_list)

    #         # load gt labels
    #         if img_type == 'good':
    #             y.extend([0] * len(img_fpath_list))
    #             mask.extend([None] * len(img_fpath_list))
    #         else:
    #             y.extend([1] * len(img_fpath_list))
    #             gt_type_dir = os.path.join(gt_dir, img_type)
    #             img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
    #             gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
    #                              for img_fname in img_fname_list]
    #             mask.extend(gt_fpath_list)

    #     assert len(x) == len(y), 'number of x and y should be same'

    #     return list(x), list(y), list(mask)

    def load_dataset_folder(self):
        phase = 'D43T' if self.is_train else 'D4NG'
        x, y, mask = [], [], []

        data_dir = os.path.join(self.dataset_path, 'data')
        csv_file = os.path.join(self.dataset_path, 'One-class-D4NG-D43T', 'D4NG.csv')
        # gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')

        if self.is_train:
            # load train dataset
            img_dir = os.path.join(data_dir, phase)
            img_fpath_list = sorted(
                    [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')])
            x.extend(img_fpath_list[:])
            y.extend([0 for _ in range(len(img_fpath_list[:]))])
        else:
            # load test dataset
            img_dir = os.path.join(data_dir, phase)
            with open(csv_file, 'r') as read_obj:
                csv_reader = csv.reader(read_obj)
                for row in csv_reader:
                    x.append(os.path.join(img_dir, row[0]))
                    y.append(1)
            img_dir = os.path.join(data_dir, 'D43T')
            img_fpath_list = sorted(
                    [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')])
            for img in img_fpath_list[:50]:
                x.append(img)
                y.append(0)

        mask = [torch.zeros([1, self.resize, self.resize]) for _ in range(len(x))]
        assert len(x) == len(y), 'number of x and y should be same'
        
        return list(x), list(y), list(mask)



