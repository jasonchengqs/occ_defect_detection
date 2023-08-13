import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import Dataset, TensorDataset, DataLoader
from folderFilter import ImageFolderFilter, ImageFolderPickle
from torch import optim
import torch.nn as nn
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import argparse

# BATCH_SIZE = 144
# DATA_PATH = "/mnt/ml/DATASET/Inspection/data/"
# num_epochs = 1


def parse_args():
    """Parser for command line arguments.
    """
    def _bool(arg):
        ua = str(arg).upper()
        if 'TRUE'.startswith(ua):
            return True
        elif 'FALSE'.startswith(ua):
            return False
        else:
            pass
    
    parser = argparse.ArgumentParser(description='Prepare datasets.')
    # required arguments
    parser.add_argument('--data-path', type=str, default='',
                        help='absolute path of the raw, unpreprocessed dir.')
    parser.add_argument('--save-path', type=str, default='',
                        help='absolute path of the save dir.')
    opt = parser.parse_args()
    return opt

# data loader
def get_dataloader(data_path, selected_class, target_transform):
	# Todo add various randomness, e.g. rotation, noise, etc.
	TRANSFORM_IMG = transforms.Compose([
	    transforms.Resize(256),
	    transforms.CenterCrop(224),
		transforms.RandomHorizontalFlip(0.5),
		transforms.RandomVerticalFlip(0.5),
	    transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406],
		                     std =[0.229, 0.224, 0.225])
	    ])
	# todo add some noise to image for training
	all_dataset = ImageFolderFilter(
		root=data_path, 
		selected_class=selected_class, 
		transform=TRANSFORM_IMG,
		target_transform=target_transform)
	return all_dataset

if __name__ == '__main__':
	
	SELECTED_CLASS = {
        'D6DRF': 0,
        'D6BU': 1,
        'D5OF': 2,
        'D4NG': 3, 'D6XC': 3, 'D6XE1': 3, 'D6XE2': 3, 
        'D6XE3': 3, 'D6XE4': 3, 'D6XE5': 3, 'D6XE6': 3,
        'D41I': 4, 'D42I': 4, 'D43T': 4, 'D45O': 4, 
        'D46O': 4, 'D6ILC': 4, 'D6ILE1': 4, 'D6ILE2': 4, 
        'D6ILE5': 4, 'D6ILE6': 4, 'D6ILSD': 4, 'D6IWC': 4, 
        'D6IWE1': 4, 'D6IWE2': 4, 'D6IWE6': 4, 'D6OL2C': 4,
        'D6OL2E1': 4, 'D6OL2E2': 4, 'D6OL2E4': 4, 'D6OL2E6': 4, 
        'D6OLC': 4, 'D6OLE1': 4, 'D6OLE2': 4, 'D6OLE4': 4, 
        'D6OLE5': 4, 'D6OLE6': 4, 'D6OWC': 4, 'D6OWE1': 4, 
        'D6OWE2': 4, 'D6OWE3': 4, 'D6OWE4': 4, 'D6OWE5': 4, 
        'D6OWE6': 4, 'D6TOC': 4, 'D6TOE1': 4, 'D6TOE2': 4, 
        'D6TOE3': 4, 'D6TOE4': 4, 'D6TOE5': 4, 'D6TOE6': 4}

	TARGET_TRANSFORM = {0:0, 1:0, 2:0, 3:0, 4:1}

	opt = parse_args()

	all_dataset = get_dataloader(
		data_path=opt.data_path, 
		selected_class=SELECTED_CLASS, 
		target_transform=TARGET_TRANSFORM)
	
	save_file_path = opt.save_path
	all_dataset.get_stratify_splits(save_file_path)