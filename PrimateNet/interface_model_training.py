import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data as utils
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import numpy as np

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--id', type=int)


traindir = "/home/zly27/ImageNet/"
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
dataset = datasets.ImageFolder(traindir, transforms.Compose([
											transforms.RandomResizedCrop(224),
											transforms.RandomHorizontalFlip(),
											transforms.ToTensor(),
											normalize,
										 ]))

print(dataset.class_to_idx)

train_loader = torch.utils.data.DataLoader(
	dataset, batch_size = 100, shuffle=True, num_workers=1)

K = 18

G = []
for i in range(K):
	G.append([])
for i, (X, y) in enumerate(train_loader):
	for j in range(len(y)):
		idx = int(y[j])
		G[idx].append(X[j].reshape(1, 3, 224, 224))
	print(i)


import pickle

args = parser.parse_args()

idx = args.id

print("Now ... %d" % (idx))

if (idx == 18): lst = [0, 1, 2]
elif (idx == 19): lst = [3, 4]
elif (idx == 20): lst = [5,6,7,8,9]
elif (idx == 21): lst = [10, 11, 12, 13, 14, 15]
elif (idx == 22): lst = [16, 17]
elif (idx == 23): lst = [0,1,2,3,4]
elif (idx == 24): lst = [5,6,7,8,9,10,11,12,13,14,15]
else: lst = [idx]
trainX, trainy = [], []


for i in range(K):
	if (i in lst): 	
		X = np.concatenate(G[i])[:1000]
		y = np.zeros(X.shape[0])
	else: 
		X = np.concatenate(G[i])[:1000]
		y = np.ones(X.shape[0])
	trainX.append(X)
	trainy.append(y)


trainX = np.concatenate(trainX)
trainy = np.concatenate(trainy)

trainX = torch.from_numpy(trainX)
trainy = torch.from_numpy(trainy).type(torch.LongTensor)

print(trainX.size())
dataset = utils.TensorDataset(trainX, trainy)
trainloader = utils.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)

testX, testy = [], []

for i in range(K):
	if (i in lst): 	
		X = np.concatenate(G[i])[1000:1300]
		y = np.zeros(X.shape[0])
	else: 
		X = np.concatenate(G[i])[1000:1300]
		y = np.ones(X.shape[0])
	#X = np.concatenate(G[idx])[1000:1300]
	
	#if (i == idx): y = np.zeros(X.shape[0])
	#else: y = np.ones(X.shape[0])
	testX.append(X)
	testy.append(y)

testX = np.concatenate(testX)
testy = np.concatenate(testy)

testX = torch.from_numpy(testX)
testy = torch.from_numpy(testy).type(torch.LongTensor)
print(testX.size())
dataset = utils.TensorDataset(testX, testy)
testloader = utils.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)

import net
from net import Net
from resnet import resnet18
net2 = resnet18(pretrained=True)
net2 = nn.DataParallel(net2)

net2.load_state_dict(torch.load("premodel.pt"))

net2 = net2.cuda()
net2.eval()

net = Net()
net = net.cuda()

criterion = nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
noise_sd = 0.50
optimizer = optim.Adam(net.parameters(), lr = 2e-4, betas=(0.5, 0.999), weight_decay=2e-4)
for epoch in range(80):
	total = 0
	correct = 0
	for i, data in enumerate(trainloader, 0):
		X, y = data
		X = X.cuda()
		y = y.cuda()
		noises = torch.randn_like(X, device=device) * noise_sd
		optimizer.zero_grad()
		output = net(net2(X + noises).view(-1, 512))
		loss = criterion(output, y)
		loss.backward()
		optimizer.step()
		total += y.size(0)

		_, predicted = torch.max(output.data, 1)
		correct += (predicted == y).sum().item()
		if (i % 100 == 0):
			print("Epoch %d: [%d/%d]: %.2f" % (epoch, total, 50000, 100. * correct / total))
	total = 0
	correct = 0
	correctp = 0
	totalp = 0
	for i, data in enumerate(testloader, 0):
		X, y = data
		X = X.cuda()
		y = y.cuda()

		noises = torch.randn_like(X, device=device) * noise_sd
		output = net(net2(X + noises).view(-1, 512))
		_, predicted = torch.max(output.data, 1)
		total += y.size(0)
		correct += (predicted == y).sum().item()
		correctp += ((predicted == y) * y).sum().item()
		totalp += (y == 1).sum().item()

	print('Test Accuracy %.2f' % (100. * correct / total))
	print('0 Test Accuracy %.2f' % (100. * correctp / totalp))
	
	print('1 Test Accuracy %.2f' % (100. * (correct - correctp) / (total - totalp)))
	#scheduler.step()

print(net)
torch.save(net.state_dict(), "sigma%.2f/submodel%d.pt" % (noise_sd, idx))

