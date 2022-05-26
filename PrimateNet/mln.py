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
import pickle

N = 25
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mat = []
dist = np.zeros(N)
dist2 = np.zeros(N)
edge = []
for i in range(25):
	edge.append([])

edge[18] = [0,1,2]
edge[19] = [3,4]
edge[20] = [5,6,7,8,9]
edge[21] = [10,11,12,13,14,15]
edge[22] = [16, 17]
edge[23] = [18,19]
edge[24] = [20, 21]

w = 2
def p2w(prob):
	return np.log(prob) - np.log(1. - prob)

def dfs(x, fa, cur):
	dist[x] = p2w(mat[cur][x])
	#else: dist[x] = 0
	if (fa != -1): dist[x] += dist[fa];
	for i in range(len(edge[x])):
		dfs(edge[x][i], x, cur)

def dfs2(x, fa, cur):
	dist2[x] = dist[x]
	#else: dist[x] = 0
	for i in range(len(edge[x])):
		dfs2(edge[x][i], x, cur)
		dist2[x] += dist2[edge[x][i]]

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

from resnet import resnet18
net2 = resnet18(pretrained=True)
net2 = nn.DataParallel(net2)

net2.load_state_dict(torch.load("premodel.pt"))

net2 = net2.cuda()
net2.eval()
noise_sd = 0.5

testX = []
testy = []
for i in range(K):

	for j in range(30):
		X = np.concatenate(G[i])[1000+j*10:1010+j*10]
		y = np.ones(X.shape[0]) * i
		X = torch.from_numpy(X).cuda()	
		noises = torch.randn_like(X, device=device) * noise_sd
		testX.append(net2(X + noises).view(-1, 512).cpu().detach().numpy())
		testy.append(y)



testX = torch.from_numpy(np.concatenate(testX))
testy = torch.from_numpy(np.concatenate(testy)).type(torch.LongTensor)

print(testX.shape)
print(testy)

#with open("inter.pt", "rb") as tf:
#	(testX, testy) = pickle.load(tf)
models = []

import net
from net import Net

for i in range(N):
	models.append(Net())
	models[i].load_state_dict(torch.load("sigma0.50/submodel%d.pt" % (i)))
	models[i] = models[i].cuda()

# w/o knowledge
mat = []
for i in range(N):
	testX = testX.cuda()
	output = (models[i](testX)[:,0].data).cpu().numpy()
	prob = 1. - 1. / (1 + np.exp(output))
	mat.append(prob.reshape(output.shape[0], 1))

mat = np.concatenate(mat, axis=1)
print(mat.shape)

correct, wcorrect = 0, 0

alpha = 0.0
sz = max(1, min(N, int(alpha * N)))
eps = 0.0
attacked = np.random.permutation(N)[:sz]
cnt = 0
#attacked = [8]
fa = np.zeros(N)
for i in range(N): fa[i] = -1

for j in range(N):
	for k in range(len(edge[j])):
		fa[edge[j][k]] = int(j)

for i in range(mat.shape[0]):

	for j in range(len(attacked)):
		x = mat[i][attacked[j]]
		if (x < 0.5): mat[i][attacked[j]] = np.clip(x + eps, 1e-5, 1 - 1e-5)
		else: mat[i][attacked[j]] = np.clip(x - eps, 1e-5, 1 - 1e-5)

	truth = np.zeros(N)
	j = int(testy[i])
	while (True):
		#print(j)
		truth[j] = 1
		j = int(fa[j])
		if (j == -1): break
	
	for j in range(N):
		pred = (mat[i][j] >= 0.5)
		correct += (pred == truth[j])
	cnt += 1	
	dist[22], dist[23], dist[24] = 0, 0, 0
	dfs(22, -1, i)
	dfs(23, -1, i)
	dfs(24, -1, i)
	tmps = 0

	for j in range(N):
		dist[j] = np.exp(dist[j] + tmps)
	
	dfs2(22, -1, i)
	dfs2(23, -1, i)
	dfs2(24, -1, i)
	Sum = dist2[22] + dist2[23] + dist2[24]
	

	for j in range(N):
		pred = (dist2[j] * 2 >= Sum)
		wcorrect += (pred == truth[j])


print("w/ knoeldge %.4f, w/o knowledge %.4f" % (wcorrect * 1. / (cnt * N), correct * 1. / (cnt * N)))
