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
import time
import shutil
from Meters import *
import argparse

BATCH_SIZE = 400
DATA_PATH = "/mnt/ml/DATASET/Inspection/data/"
num_epochs = 100


def set_parameter_require_grad(model, feature_extraction):
	if feature_extraction:
		for param in model.parameters():
			param.require_grad = False


def get_ImageNet_model(num_classes=5):
	from resnet import wide_resnet50_2, wide_resnet101_2
	model = wide_resnet50_2(pretrained=True)
	num_ftrs = model.fc.in_features
	model.fc = nn.Linear(num_ftrs, num_classes)
	return model


def train(dataloader, model, criterion, optimizer, epoch, device):
	batch_time = AverageMeter('Time', ':6.3f')
	data_time = AverageMeter('Data', ':6.3f')
	losses = AverageMeter('Loss', ':.4e')
	top1 = AverageMeter('Acc@1', ':6.2f')
	progress = ProgressMeter(
		len(dataloader),
		[batch_time, data_time, losses, top1],
		prefix="Epoch: [{}]".format(epoch))
	
	model.train()
	end = time.time()
	for i, (names, x, y) in enumerate(dataloader):
		data_time.update(time.time() - end)
		
		optimizer.zero_grad()
		x = x.to(device)
		y = y.to(device)
		y_hat, _ = model(x)
		loss = criterion(y_hat, y)
		loss.backward()
		optimizer.step()
		
		acc1 = accuracy(y_hat, y, topk=(1,))
		losses.update(loss.item(), x.size(0))
		top1.update(acc1[0], x.size(0))
		
		batch_time.update(time.time() - end)
		end = time.time()
		
		# if i % 1000 == 0:
		# progress.display(i)
	
	return model


def validate(dataloader, model, criterion, device, filename='predicted_results.pkl'):
	batch_time = AverageMeter('Time', ':6.3f')
	losses = AverageMeter('Loss', ':.4e')
	top1 = AverageMeter('Acc@1', ':6.2f')
	progress = ProgressMeter(
		len(dataloader),
		[batch_time, losses, top1],
		prefix='Test: ')
	
	# switch to evaluate mode
	model.eval()
	classwise_acc = {}
	sample_wise_info = {}
	all_result = {'target': [], 'output': []}
	for i in range(5):
		classwise_acc[i] = [0, 0]
	with torch.no_grad():
		end = time.time()
		for i, (names, x, y) in enumerate(dataloader):
			x = x.to(device)
			y = y.to(device)
			y_hat, _ = model(x)
			# compute output
			loss = criterion(y_hat, y)
			
			# measure accuracy and record loss
			classwise_acc, all_result = cal_classwise_acc(y_hat, y, classwise_acc, all_result)
			acc1 = accuracy(y_hat, y, topk=(1,))
			losses.update(loss.item(), x.size(0))
			top1.update(acc1[0], x.size(0))
			
			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()
			
			class_names = y.cpu().numpy()
			predict_names = y_hat.cpu().numpy()
			
			for name, y_hat_, y_ in zip(names, predict_names, class_names):
				sample_wise_info[name] = [y_, y_hat_]
			
			# if i % 1000 == 0:
			# progress.display(i)
		
		# TODO: this should also be done with the ProgressMeter
		print(top1.avg)
	pickle.dump(all_result, open(filename, 'wb'))
	pickle.dump(sample_wise_info, open('detailed_' + filename, 'wb'))
	return top1.avg, classwise_acc, sample_wise_info


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
	torch.save(state, filename)
	if is_best:
		shutil.copyfile(filename, 'model_best.pth.tar')


def parse_args():
	"""
	Parser for inline arguments.
	"""
	parser = argparse.ArgumentParser(description='Do Training on ResNet.')
	# required arguments
	parser.add_argument('--data_path', type=str, default='./train_test_set1.pkl',
	                    help='data_path')
	opt = parser.parse_args()
	config = vars(opt)
	return config


if __name__ == '__main__':
	config = parse_args()
	DATA_PATH = config['data_path']
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = get_ImageNet_model().to(device)
	
	TRANSFORM_IMG = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.RandomHorizontalFlip(0.5),
		transforms.RandomVerticalFlip(0.5),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406],
		                     std=[0.229, 0.224, 0.225])
	])
	train_set = ImageFolderPickle(DATA_PATH, True, transform=TRANSFORM_IMG)
	train_data_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
	
	TRANSFORM_IMG_TEST = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406],
		                     std=[0.229, 0.224, 0.225])
	])
	test_set = ImageFolderPickle(DATA_PATH, False, transform=TRANSFORM_IMG_TEST)
	test_data_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
	
	# criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0,0.20,0.111,0.05])).to(device)
	# criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 0.08])).to(device)
	
	criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0,0.20,0.110,0.05,0.0013])).to(device)
	
	# criterion = nn.CrossEntropyLoss().to(device)
	optimizer = torch.optim.SGD(model.parameters(), 0.001)
	
	if True:
		check_point_model = None
		pass
	best_acc = 0
	for epoch in range(num_epochs):
		train(train_data_loader, model, criterion, optimizer, epoch, device)
		acc, classwise_acc, _ = validate(test_data_loader, model, criterion, device,
		                                 'predicted_results_5cls' + str(DATA_PATH.split('.')[1][-8]) + '.pkl')
		is_best = acc > best_acc
		best_acc1 = max(acc, best_acc)
		
		save_checkpoint({
			'epoch': epoch + 1,
			'state_dict': model.state_dict(),
			'best_acc1': best_acc1,
			'optimizer': optimizer.state_dict(),
		}, is_best, filename='checkpoint_split' + str(DATA_PATH.split('.')[1][-1]) + '.pth.tar')
		print(classwise_acc)