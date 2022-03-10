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
import warnings
from decimal import *
from model2 import NEURAL as NEURAL_main
from model import NEURAL as NEURAL_attr

from statsmodels.stats.proportion import proportion_confint
#warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--sigma', type=float)
parser.add_argument('--r', type=float)
args = parser.parse_args()

N = 22 # [0, 12) => main sensor bool var, [12, 25) => attr sensor bool var

def p2w(prob): # w_G = log(p/(1-p))
	prob = np.clip(prob, 1e-6, 1 - 1e-6) # save for some precision overfloat
	return np.log(prob) - np.log(1. - prob)
	#else: 
	#	if (prob < 0.5): return np.log(prob) - np.log(1 - prob) + 2
	#	else: return np.log(prob) - np.log(1 - prob)

	#return np.log(prob) - 2 * np.log(1 - prob)
def trans(pA, delta): # pA => \Phi^{-1}(\Phi(pA) - r / sigma)
	return norm.cdf(norm.ppf(pA) - delta)

	
def output_prob(_interface):
	weight_of_all_worlds = torch.zeros(_interface.shape[0])
	weight_of_worlds_contains_me = torch.zeros(_interface.shape[0], 12) #Decimal(0), Decimal(0)
	for i in range(len(rules)):
		current_world_weight = torch.zeros(_interface.shape[0])
		for j in range(len(rules[i])):
			if (rules[i][j]): current_world_weight += p2w(_interface[:,j])
		current_world_weight = np.exp(current_world_weight)
		weight_of_all_worlds += current_world_weight
		for j in range(12):
			if (rules[i][j] == True): weight_of_worlds_contains_me[:,j] += current_world_weight

	for i in range(_interface.shape[0]): 
		for j in range(12):
			weight_of_worlds_contains_me[i][j] /= weight_of_all_worlds[i]

	return weight_of_worlds_contains_me

with open("inter.pt", "rb") as tf: # Data loader
	(testX, testy) = pickle.load(tf)
with open("newrules.pt", "rb") as tf: # Possible worlds
	rules = pickle.load(tf)
main_model = NEURAL_main(n_class=12, n_channel=3)
main_model.load_state_dict(torch.load("main/model_%.2f.pt" % (args.sigma)))
main_model.cuda()
main_model.eval()

attr = []
for i in range(13):
	model = NEURAL_attr(n_class = 1, n_channel=3)
	if (i == 3 or i == 7 or i == 8): continue
	if (i == 1 or i == 7 or i == 9):
		model.load_state_dict(torch.load("tpenhance/model_%d_%.2f_10.pt" % (i, args.sigma)))
	elif (i == 3 or i == 5):
		model.load_state_dict(torch.load("tnenhance/model_%d_%.2f_1.pt" % (i, args.sigma)))
	elif (i == 0 or i == 12):
		model.load_state_dict(torch.load("nc/model_%d_%.2f_5.pt" % (i, args.sigma)))
	else:
		model.load_state_dict(torch.load("nc/model_%d_%.2f.pt" % (i, args.sigma)))
	model = model.cuda()
	model = model.eval()
	attr.append(model)

rules = np.concatenate([rules[:,:12+3], rules[:,12+4:12+7], rules[:,12+9:]], axis=1)
#mat = np.concatenate([mat[:,:12+3], mat[:,12+4:12+7], mat[:,12+9:]], axis=1)

#rules = np.concatenate([rules[:,:12+4], rules[:,12+4:12+7], rules[:,12+11:]], axis=1)
#mat = np.concatenate([mat[:,:12+4], mat[:,12+4:12+7], mat[:,12+11:]], axis=1)

#print(rules.shape)
#print(mat.shape)
#print(rules)
eps = args.r / args.sigma 

global all_perturbed_sensors
global sampled_lambda

#

all_perturbed_sensors = [i for i in range(N)] # all sensors are perturbed

all_cases = 0
#for i in range(mat.shape[0]): all_cases += (mat[i][testy[i].item()] >= 0.5)

related_worlds = [[] for i in range(N)]
for i in range(len(rules)):
	for j in range(len(rules[i])):
		if (rules[i][j]): related_worlds[j].append(i)

tp, A, w_A = 0, 0, 0
b_a, b_b = 0, 0
softmax = nn.Softmax(dim=1)

def _count_arr(arr, length):
	counts = np.zeros(12, dtype=int)
	for idx in arr:
		counts[idx] += 1
	return counts


pas = []

print("sigma = %.2f" % (args.sigma))
print("idx\tpA\tvar")

for i in range(testX.shape[0]):
	#print(testX[i].shape)
	X = torch.from_numpy(testX[i])
	X = X.cuda()
	pos = testy[i].item()

	#X = X + torch.randn_like(X).cuda() * args.sigma
	#main_p = softmax(main_model(X)).squeeze()
	#interface = [main_p[j].item() for j in range(12)]
	#for j in range(10): interface.append((softmax(attr[j](X)).squeeze())[1].item())

	num = 100000
	N0 = 100000
	batch_size = 6000
	alpha = 0.001
	with torch.no_grad():
		counts = []
		for _ in range(ceil(num / batch_size)):
			this_batch_size = min(batch_size, num)
			num -= this_batch_size
			batch = X.repeat((this_batch_size, 1, 1, 1))
			noise = torch.randn_like(batch, device='cuda') * args.sigma

			main_p = softmax(main_model(batch + noise)).squeeze().cpu().detach().numpy()

			interface = [main_p[:, j].reshape(-1, 1) for j in range(12)]
			for j in range(10): interface.append((softmax(attr[j](batch + noise)))[:,1].cpu().detach().numpy().reshape(-1, 1))
			interface = np.concatenate(interface, axis=1)
			predictions = output_prob(interface)
			counts.append(predictions[:,pos].squeeze().numpy())
	
	counts = np.concatenate(counts)
	if (np.mean(counts) < 0.5): continue
	print("%d\t%.3f\t%.3f" % (tp, np.mean(counts), np.var(counts)))
	#pa = proportion_confint(counts[testy[i].item()].item(), N0, alpha=2*alpha,method="beta")[0]
	#pas.append(pa)
	#pos = testy[i].item() # Certify the #ground-truth dimension >= 0.5
	
	
	#print(interface[pos])
	#if (interface[pos] >= 0.5): A += 1
	#if (_ / __ >= 0.5): w_A += 1
	#tp += 1
	#if (interface[pos] >= 0.5 and _ / __ < 0.5):
	#print("[%d] Interface confidence: %.5f" % (pos, interface[pos]))
	#print("Marginal confidence before perturbation: %.5f" % (_ / __))
	
	#if (interface[pos] >= 0.5): A += 1
	#if (_ / __ >= 0.5): w_A += 1
	#	ta = "[Main] "
	#	for j in range(12):
	#		ta += " Sensor %d: %.5f" % (j, interface[j])
	#	print(ta)
	#	ta = "[Attr] "
	#	for j in range(12, N):
	#		ta += " Sensor %d: %.5f" % (j - 12, interface[j])
	#
	#	print(ta)
	#continue
	#per = np.zeros(N)
	#sampled_lambda = []
	#min_prob, max_prob = 0, 1
	#recursion(0, 0, 0, np.sum(np.exp(world_weight_a)), np.sum(np.exp(world_weight_b)))

	#without_knowledge = [trans(interface[pos], eps), trans(interface[pos], -eps)]
	#print("[Theoretical bound w/o knowledge] Min: %.5f, Max: %.5f" % (without_knowledge[0], without_knowledge[1]))
	#print("[Theoretical bound w/ knowledge] Min: %.5f, Max: %.5f" % (min_prob, max_prob))

	#if (without_knowledge[0] >= 0.5): A += 1
	#if (without_knowledge[1] <= 0.5 and xflag == False): A += 1

	#if (min_prob >= 0.5): w_A += 1
	#if (max_prob <= 0.5 and xflag == False): w_A += 1

	tp += 1
	if (tp == 200): break
	#if (tp % 10 == 0):
	#	print("[%d/%d] w/o knowledge: -, w/ knowledge: -" % (tp, testX.shape[0]))# % (tp,testX.shape[0], 100 * A / tp, 100 * w_A / tp))
	#	with open("pa_%.2f.pkl" % (args.sigma), "wb") as tf:
	#		pickle.dump(pas, tf)
#print("w/o knowledge: %.4f, w/ knowledge: %.4f" % (100 * A / tp, 100 * w_A / tp))
