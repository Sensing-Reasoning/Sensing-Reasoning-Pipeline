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
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import numpy as np
traindir = "/home/zly27/ImageNet/"
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
dataset = datasets.ImageFolder(traindir, transforms.Compose([
											transforms.RandomResizedCrop(224),
											transforms.RandomHorizontalFlip(),
											transforms.ToTensor(),
											normalize,
										 ]))

print(dataset.class_to_idx)
import pickle
from resnet import resnet18
net2 = resnet18(pretrained=True)


net2 = nn.DataParallel(net2)
net2 = net2.cuda()
net2.eval()
torch.save(net2.state_dict(), "premodel.pt")
'''
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


for i in range(2):
	print(len(G[i]))
	X = np.concatenate(G[i])
	
#	X = torch.from_numpy(X)
#	X = X.cuda()
#	tx = []
#	for j in range(13):
#		x = X[j*100:(j+1)*100]
#		xx = net2(x)
#		xx = xx.cpu().detach().numpy().reshape(-1, 512)
#		tx.append(xx)
#	print(xx.shape)
#	tx = np.concatenate(tx)
	print(X.shape)
	with open("rdata_%d.pkl" % (i), "wb") as tf:
		pickle.dump(X[:1000], tf)
	with open("rtestdata_%d.pkl" % (i), "wb") as tf:
		pickle.dump(X[1000:1300], tf)
'''
