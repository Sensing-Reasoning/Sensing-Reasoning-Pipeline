import argparse
import os
import random
import shutil
import time
import warnings
from math import *
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
from scipy.stats import norm
import numpy as np
import pickle
global Sum
global pos
global xflag
global XX
global Z1_max
global Z2_min
global min_prob
global max_prob


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--sigma', type=float)
#parser.add_argument('--id', type=int)
args = parser.parse_args()
N = 12

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mat = []

#dist = np.zeros(N)
#dist2 = np.zeros(N)

#edge = []
#for i in range(25):
#	edge.append([])

#edge[18] = [0,1,2]
#edge[19] = [3,4]
#edge[20] = [5,6,7,8,9]
#edge[21] = [10,11,12,13,14,15]
#edge[22] = [16, 17]
#edge[23] = [18,19]
#edge[24] = [20, 21]

truth = np.array([
[1,0,0,0,0,1,0,0,0,0,0,0]
,[0,1,0,0,0,0,0,1,0,0,0,0]
,[0,0,1,0,1,1,0,0,0,0,0,0]
,[0,0,1,0,0,1,0,0,0,0,0,0]
,[0,0,0,1,0,0,1,0,0,0,0,1]
,[0,0,0,1,0,0,1,0,0,0,0,0]
,[0,0,0,1,0,1,0,0,0,0,0,0]
,[0,0,0,1,1,1,0,0,0,0,0,0]
,[0,0,0,1,0,1,0,0,1,0,0,0]
,[0,0,0,1,0,0,0,1,0,0,0,0]
,[0,0,0,1,0,1,0,0,0,1,0,0]
,[0,0,0,1,0,1,0,0,0,0,1,0]])

w = 2
def p2w(prob):
	prob = np.clip(prob, 1e-20, 1 - 1e-20)
	return np.log(prob) - np.log(1. - prob)

#def dfs(x, fa, rc, depth):
#	global XX
#	dist[x] = np.exp(p2w(rc[x]))# * (2 ** depth)## - p2w(XX[x])#p2w(1. - rc[x])
	#else: dist[x] = 0
#	if (fa != -1): dist[x] = dist[x] * dist[fa] + dist[x] * dist[fa] / np.exp(p2w(rc[fa]));
#	for i in range(len(edge[x])):
#		dfs(edge[x][i], x, rc, depth + 1)

#def dfs2(x, fa, rc, depth):
#	#dist2[x] = dist[x]
#	if (x <= 17): dist2[x] = dist[x]
#	else: dist2[x] = 0
#	#else: dist[x] = 0
#	for i in range(len(edge[x])):
#		dfs2(edge[x][i], x, rc, depth + 1)
#		if (edge[x][i] <= 17): 
#			dist2[x] += (dist2[edge[x][i]] * rc[x])## 2**(depth+1) possiblilty for child
#		else:
#			dist2[x] += dist2[edge[x][i]] / rc[edge[x][i]] * rc[x]

from dataset import DataMain
from model2 import NEURAL
print('[Data] Done .... ')
def trans(y, delta):
	return norm.cdf(norm.ppf(y) - delta)

#models = []
#for i in range(N):

#	models[i] = models[i].cuda()
#	models[i] = models[i].eval()

# w/o knowledge

with open("inter.pt", "rb") as tf:
	(testX, testy) = pickle.load(tf)

testX = torch.from_numpy(testX)
testy = torch.from_numpy(testy)
#mat = []
def _count_arr(arr, length):
	counts = np.zeros(12, dtype=int)
	for idx in arr:
		counts[idx] += 1
	return counts

def _sample_noise(args, model, x, num, batch_size):
	with torch.no_grad():
		counts = np.zeros(12, dtype=int)
		for _ in range(ceil(num / batch_size)):
			this_batch_size = min(batch_size, num)
			num -= this_batch_size
			batch = x.repeat((this_batch_size, 1, 1, 1))
			noise = torch.randn_like(batch, device='cuda') * args.sigma
			predictions = model(batch + noise).argmax(1)
			counts += _count_arr(predictions.cpu().numpy(), 12)
	return counts


from statsmodels.stats.proportion import proportion_confint

def certify(args, model, X, N0=100, N=100000, alpha=0.001, batch_size=6000):
	#count = _sample_noise(args, model, X, N0, batch_size)
	#ca = count.argmax().item()
	pAs = []
	est = _sample_noise(args, model, X, N, batch_size)
	#nA = est[ca].item()
	for i in range(12):
		pa = proportion_confint(est[i].item(), N, alpha=2*alpha,method="beta")[0]
		pAs.append(pa)
	return np.array(pAs)

testX = testX.cuda()

certified = 0

#for i in range(13):
model = NEURAL(n_class=12,n_channel=3)
model.load_state_dict(torch.load("main/model_%.2f.pt" % (args.sigma)))
model = model.cuda()
model = model.eval()
tmpm = []
for j in range(testX.shape[0]):
	output = certify(args, model, testX[j:(j+1)])#(models[i](testX)[:,0].data).cpu().numpy()
	if (j % 10 == 9): 
		print("[%d/%d]" % (j + 1, testX.shape[0]))
	tmpm.append(output.reshape(1, 12))
tmpm = np.concatenate(tmpm,axis=0)

#tmpm = np.array(tmpm)
#mat.append(tmpm.reshape(-1, 1))

#print(100. * certified / testX.shape[0])
#mat = np.concatenate(mat, axis=1)
#print(mat.shape)

print(tmpm.shape)
with open("main_%.2f.pkl" % (args.sigma), "wb") as tf:
	pickle.dump(tmpm, tf)

'''
with open("mat_%.2f.pkl" % (args.sigma), "rb") as tf:
	mat = pickle.load(tf)

print(mat.shape)
print(mat[0], testy[0])
print(mat.shape)

correct, wcorrect = 0, 0


#cnt = 0
#attacked = [8]
#fa = np.zeros(N)
#for i in range(N): fa[i] = -1

#for j in range(N):
#	for k in range(len(edge[j])):
#		fa[edge[j][k]] = int(j)

from copy import deepcopy



def dfs4(cur, per, per2, xx, yy, lam, lst, flag):
	global XX
	global Z1_max
	global Z2_min
	global Z3_max
	global Z3_min
	global pos
	global xflag
	if (cur == N):
		XXX = deepcopy(XX)
		for i in range(N):
			if (xx[i] != ' '):
				y = XXX[i]
				#print(i, y)
				if (y >= 0.5):
					y = np.clip(min(y, trans(y, flag * yy[i])), 1e-5, 1 - 1e-5)
				else:
					y = np.clip(max(y, trans(y, flag * yy[i])), 1e-5, 1 - 1e-5)
				#else:
				#print(i, y)
				XXX[i] = y

		se = 0
		for i in range(len(lam)):
			la = lam[i]
			d1 = XXX[lst[i]]
			d2 = XX[lst[i]]
			delta = p2w(d1) - p2w(d2)
			se += la * delta
		#print("Z1")
		#old, new = "", ""
		#for i in range(N):
		#	old += "%.5f\t" % (XX[i])
		#	new += "%.5f\t" % (XXX[i])
		#print(old)
		#print(new)
		Z3 = output_prob(pos, xflag, XXX)
		z1 = Z3 * np.exp(se)
		#print(z1)
		XXX = deepcopy(XX)
		for i in range(N):
			if (yy[i] != ' '):
				y = XXX[i]
				if (y >= 0.5):
					y = np.clip(min(y, trans(y, flag * xx[i])), 1e-5, 1 - 1e-5)
				else:
					y = np.clip(max(y, trans(y, flag * xx[i])), 1e-5, 1 - 1e-5)
				XXX[i] = y
		se = 0
		for i in range(len(lam)):
			la = lam[i]
			d1 = XXX[lst[i]]
			d2 = XX[lst[i]]
			delta = p2w(d1) - p2w(d2)
			se += la * delta
		#print("Z2")
		#old, new = "", ""
		#for i in range(N):
		#	old += "%.5f\t" % (XX[i])
		#	new += "%.5f\t" % (XXX[i])
		#print(old)
		#print(new)
		output_prob(pos, xflag, XXX)
		Z2 = Sum
		z2 = Z2 * np.exp(se)
		#print(z2)
		if (Z1_max == None):
			Z1_max = z1
		else: 
			if (flag == -1): Z1_max = min(Z1_max, z1)
			else: Z1_max = max(Z1_max, z1)

		if (Z2_min == None):
			Z2_min = z2
		else: 
			if (flag == -1): Z2_min = max(Z2_min, z2)
			else: Z2_min = min(Z2_min, z2)
		return 
	if (len(per[cur]) == 0):
		xx.append(' ')
		yy.append(' ')
		dfs4(cur + 1, per, per2, xx, yy, lam, lst, flag)
	elif (len(per[cur]) == 1):
		xx.append(per[cur][0])
		yy.append(per2[cur][0])
		dfs4(cur + 1, per, per2, xx, yy, lam, lst, flag)
	else:
		xx.append(per[cur][0])
		yy.append(per2[cur][0])
		dfs4(cur + 1, per, per2, xx, yy, lam, lst, flag)
		xx.pop()
		yy.pop()
		xx.append(per[cur][1])
		yy.append(per2[cur][1])
		dfs4(cur + 1, per, per2, xx, yy, lam, lst, flag)
	xx.pop()
	yy.pop()

		

def calc(per, per2, lam, lst):
	global Z1_max
	global Z2_min
	global min_prob
	global max_prob
	Z1_max = None
	Z2_min = None
	dfs4(0, per, per2, [], [], lam, lst, -1)
	#print(Z1_min, Z2_max)
	#print(Z1_max, Z2_min)
	#if (Z1_max / Z2_min > min_prob):
	min_prob = max(min_prob, Z1_max / Z2_min)
	return True
	#return False
	#print(Z1_max, Z2_min)
	#Z1_max = None
	#Z2_min = None
	#dfs4(0, per, per2, [], [], lam, lst, 1)
	#max_prob = min(max_prob, Z1_max / Z2_min)
	#print(min_prob)

def dfs3(cur, term, lst, lam):
	if (cur == term):
		per, per2 = [], []
		for i in range(N):
			per.append([])
			per2.append([])
		for i in range(term):
			if (lam[i] > -1 + 1e-8 and lam[i] + 1e-8 < 0):
				per[lst[i]].append(-eps)
				per[lst[i]].append(eps)

				per2[lst[i]].append(-eps)
				per2[lst[i]].append(eps)
			elif (lam[i] >= 0):
				per[lst[i]].append(eps)
				per2[lst[i]].append(-eps)
			else:
				per[lst[i]].append(-eps)
				per2[lst[i]].append(eps)
		credit = calc(per, per2, lam, lst)
		return ;
	
	for l in [-1.0000, 0.0000]:
	#for l in np.arange(-1, 1e-8, 0.5):
		#if (cur == 2 and abs(l + 1) > 1e-8): continue	
		#if (cur == 2 and abs(l + 1) > 1e-8): continue
	#for l in np.arange(-5, 5, 0.25):
		lam.append(l)
		dfs3(cur + 1, term, lst, lam)
		lam.pop()

def output_prob(pos, flag, rc):
	global Sum
	global rules
	#dist[22], dist[23], dist[24] = 0, 0, 0
	#dfs(22, -1, rc, 0)
	#dfs(23, -1, rc, 0)
	#dfs(24, -1, rc, 0)
	#tmps = 0
	#print(dist)
	#for j in range(11):
	#	tmps += p2w(1. - rc[j])
	#for j in range(N):
	#	dist[j] = np.exp(dist[j] + tmps)
	
	#dfs2(22, -1, rc, 0)
	#dfs2(23, -1, rc, 0)
	#dfs2(24, -1, rc, 0)
	#Sum = dist2[22] + dist2[23] + dist2[24]
	#print("******************")
	#print(dist2 / Sum)
	#print(Sum)
	#print(dist2[0] + dist2[1] + dist2[2] + dist2[3] + dist2[4] + dist2[5] + dist2[6])
	Sum = 0
	mysum = 0
	for i in range(len(rules)):
		tmps = 0
		for j in range(len(rules[i])):
			if (rules[i][j]): 
				tmps += p2w(rc[j])
		#print(tmps)
		tmps = np.exp(tmps)
		Sum += tmps
		if (rules[i][pos] == 1): mysum += tmps
	if (flag == 1): return mysum
	else: return Sum - mysum
with open("rules.pt", "rb") as tf:
	rules = pickle.load(tf)

alpha = args.alp
sz = min(max(1, int(alpha * N)), N)
eps = args.eps

print("Sigma = %.2f, Alpha = %.2f, Eps = %d" % (args.sigma, alpha, eps))
#attacked = np.random.permutation(11)[:sz]

A, B, w_A, w_B = 0, 0, 0, 0
tp = 0
saved = []

np.random.seed(20)

for ii in range(mat.shape[0]):
	i = ii#perm[ii]
	#print(mat[i])
	XX = deepcopy(mat[i])

	pos = -1
	colist = []
	for j in range(N):
		pred = (XX[j] >= 0.5)
		if (pred == truth[int(testy[i])][j]):
			colist.append(j)
	#print(colist)
	if (len(colist) < sz): continue
	#if (colist[0] != 1): continue
	atidx = np.random.permutation(len(colist))[:sz]
	attacked = []
	#print(atidx)
	for j in range(atidx.shape[0]):
		attacked.append(colist[atidx[j]])
	#print(attacked)
	#attacked = [2, 10, 6]
	print(attacked)	
	pos = attacked[0]
	#for j in range(len(attacked)):
	#	if (XX[attacked[j]] >= 0.5):
	#		pos = attacked[j]
	#if (pos == -1): continue
	#print(pos)
	#if (pos == -1): continue
	xflag = (XX[pos] >= 0.5)
	#print(mat[i])
	tmp = output_prob(pos, xflag, XX)

	
	#mn_prob = 1.
	#mx_prob = 0.
	
	#for epoch in range(10000):
	
	#	XX = deepcopy(mat[i])
	#	#trueeps = eps
	#	trueeps = np.random.random() * eps
	#	for j in range(len(attacked)):
	#		x = XX[attacked[j]]
			#print(x, attacked[j])
	#		if (x < 0.5): XX[attacked[j]] = np.clip(x + trueeps, 1e-5, 1 - 1e-5)
	#		else: XX[attacked[j]] = np.clip(x - trueeps, 1e-5, 1 - 1e-5)
			#print(XX[attacked[j]], attacked[j])
		#print(XX)
	#	tmp = output_prob(pos, xflag, XX)
	#	prob = tmp / Sum
	#	mn_prob = min(mn_prob, prob)
	#	mx_prob = max(mx_prob, prob)
	#print("After: Min %.9f, Max %.9f" % (mn_prob, mx_prob))
	
	XX = deepcopy(mat[i])
	typex = (XX[pos] >= 0.5)
	print(XX[pos])
	max_prob = 1
	min_prob = 0
	dfs3(0, sz, attacked, [])
	print(min_prob, max_prob)
	#el = (mn_prob - min_prob) / mn_prob
	#eu = (max_prob - mx_prob) / mx_prob
	#print("%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t" % (mn_prob, mx_prob, min_prob, max_prob, el, eu))
	XX = deepcopy(mat[i])
	if (min_prob >= 0.5 or max_prob < 0.5):
		B += 1
		#if (typex == True):
		if (min_prob >= 0.5): A += 1
	#	else:
	#		if (max_prob < 0.5): A += 1
	if (typex == True and trans(XX[pos], eps) >= 0.5):
	#	print(XX[pos])
		w_B += 1
		w_A += 1
	if (typex == False and trans(XX[pos], -eps) < 0.5):
	#	print(XX[pos])
		w_B += 1
		w_A += 1
	xi, yi = 0, 0
	
	#print(mat[i][pos])
	XX = deepcopy(mat[i])
	xi = 2. * min_prob - 1
	if (typex == True):
		yi = 2. * trans(XX[pos], eps) - 1
	#	print(min_prob, XX[pos] - eps, typex)
	else:
		yi = 1. - 2. * trans(XX[pos], -eps)
	#	print(max_prob, XX[pos] + eps, typex)
	#print(mat[i][pos])
	
	saved.append((xi, yi))
	tp += 1
	if (tp % 10 == 0):
		print(tp)
		print("w/ knowledge A: %.4f, B: %.4f, C: %.4f" % (A * 1. / tp, A * 1. / (B + 1e-16), B * 1. / tp))
		print("w/o knowledge A: %.4f, B: %.4f, C: %.4f" % (w_A * 1. / tp, w_A * 1. / (w_B + 1e-16), w_B * 1. / tp))
print("w/ knowledge A: %.4f, B: %.4f, C: %.4f" % (A * 1. / tp, A * 1. / (B + 1e-16), B * 1. / tp))
print("w/o knowledge A: %.4f, B: %.4f, C: %.4f" % (w_A * 1. / tp, w_A * 1. / (w_B + 1e-16), w_B * 1. / tp))
'''
