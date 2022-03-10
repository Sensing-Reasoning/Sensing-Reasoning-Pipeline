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

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--alp', type=float)
parser.add_argument('--sigma', type=float)
parser.add_argument('--eps', type=int)
args = parser.parse_args()
N = 12

mat = []
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

def p2w(prob):
	#prob = np.clip(prob, 1e-20, 1 - 1e-20) # save for some precision overfloat
	return np.log(prob) - np.log(1. - prob)


print('[Data] Done .... ')

with open("inter.pt", "rb") as tf:
	(testX, testy) = pickle.load(tf)


with open("mat_%.2f.pkl" % (args.sigma), "rb") as tf:
	mat = pickle.load(tf)

print(mat.shape)
print(mat[0], testy[0])
print(mat.shape)

correct, wcorrect = 0, 0

from copy import deepcopy

def trans(y, delta): return norm.cdf(norm.ppf(y) - delta)



def dfs3(cur):
	global min_prob
	global max_prob
	if (cur == len(all_perturbed_sensors)):
		##### Lower bound computation #####
		_interface = np.zeros(N)
		for i in range(N):
			if (per[i] != 0):
				#_interface[i] = trans(interface[i], per[i] * eps)
				if (interface[i] >= 0.5):
					_interface[i] = min(interface[i], trans(interface[i], per[i] * eps))
				else:
					_interface[i] = max(interface[i], trans(interface[i], per[i] * eps))
			else:
				_interface[i] = interface[i]
		old, new = "", ""
		for i in range(N):
			old += "%.5f\t" % (interface[i])
			new += "%.5f\t" % (_interface[i])
		print(old)
		print(new)
		J = 0
		for i in range(len(all_perturbed_sensors)):
			cur_lambda = sampled_lambda[i]
			cur_pos = all_perturbed_sensors[i]
			#delta = _interface[cur_pos] - interface[cur_pos]
			delta = p2w(_interface[cur_pos]) - p2w(interface[cur_pos])
			J += cur_lambda * delta
		print("Z1")
		print(np.exp(J))
		z1 = output_prob(pos, xflag, _interface)[0] * np.exp(J)

		for i in range(N):
			if (per[i] != 0):
				#_interface[i] = trans(interface[i], -per[i] * eps)
				if (interface[i] >= 0.5):
					_interface[i] = min(interface[i], trans(interface[i], -per[i] * eps))
				else:
					_interface[i] = max(interface[i], trans(interface[i], -per[i] * eps))
			else:
				_interface[i] = interface[i]
		old, new = "", ""
		for i in range(N):
			old += "%.5f\t" % (interface[i])
			new += "%.5f\t" % (_interface[i])
		print(old)
		print(new)
		J = 0
		for i in range(len(all_perturbed_sensors)):
			cur_lambda = sampled_lambda[i]
			cur_pos = all_perturbed_sensors[i]
			#delta = _interface[cur_pos] - interface[cur_pos]
			delta = p2w(_interface[cur_pos]) - p2w(interface[cur_pos])
			J += cur_lambda * delta
		print("Z2")

		print(np.exp(J))
		z2 = output_prob(pos, xflag, _interface)[1] * np.exp(J)	
		min_prob = max(min_prob, z1 / z2)
		print(z1 / z2)
		##### Upper bound computation #####
		for i in range(N):
			if (per[i] != 0):
				if (interface[i] >= 0.5):
					_interface[i] = np.clip(min(interface[i], trans(interface[i], -per[i] * eps)), 1e-5, 1-1e-5)
				else:
					_interface[i] = np.clip(max(interface[i], trans(interface[i], -per[i] * eps)), 1e-5, 1-1e-5)
			else:
				_interface[i] = interface[i]

		J = 0
		for i in range(len(all_perturbed_sensors)):
			cur_lambda = sampled_lambda[i]
			cur_pos = all_perturbed_sensors[i]
			delta = p2w(_interface[cur_pos]) - p2w(interface[cur_pos])
			J += cur_lambda * delta

		z1 = output_prob(pos, xflag, _interface)[0] * np.exp(J)

		for i in range(N):
			if (per[i] != 0):
				if (interface[i] >= 0.5):
					_interface[i] = np.clip(min(interface[i], trans(interface[i], per[i] * eps)), 1e-5, 1-1e-5)
				else:
					_interface[i] = np.clip(max(interface[i], trans(interface[i], per[i] * eps)), 1e-5, 1-1e-5)
			else:
				_interface[i] = interface[i]

		J = 0
		for i in range(len(all_perturbed_sensors)):
			cur_lambda = sampled_lambda[i]
			cur_pos = all_perturbed_sensors[i]
			delta = p2w(_interface[cur_pos]) - p2w(interface[cur_pos])
			J += cur_lambda * delta
		z2 = output_prob(pos, xflag, _interface)[1] * np.exp(J)	

		max_prob = min(max_prob, z1 / z2)
		return ;
	
	for l in [-1.0000, 0.00]:
		sampled_lambda.append(l)
		if (l >= -1e-8): per[all_perturbed_sensors[cur]] = 1
		else: per[all_perturbed_sensors[cur]] = -1
		dfs3(cur + 1)
		sampled_lambda.pop()
	
def output_prob(pos, flag, _interface):
	weight_of_all_worlds, weight_of_worlds_contains_me = 0, 0
	for i in range(len(rules)):
		current_world_weight = 0
		for j in range(len(rules[i])):
			if (rules[i][j]): current_world_weight += p2w(_interface[j])
		current_world_weight = np.exp(current_world_weight)
		weight_of_all_worlds += current_world_weight
		if (rules[i][pos] == True): weight_of_worlds_contains_me += current_world_weight
	
	if (flag == True): return weight_of_worlds_contains_me, weight_of_all_worlds
	else:
		#print((weight_of_all_worlds - weight_of_worlds_contains_me)/weight_of_all_worlds)
		#print(weight_of_all_worlds - weight_of_worlds_contains_me, weight_of_all_worlds)
		return weight_of_all_worlds - weight_of_worlds_contains_me, weight_of_all_worlds

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

global all_perturbed_sensors
global sampled_lambda

np.random.seed(20)
for i in range(mat.shape[0]):
	interface = np.zeros(N)
	for j in range(N): interface[j] = mat[i][j]
	
	pos = -1
	all_perturbed_sensors = []
	colist = []
	for j in range(N):
		pred = (interface[j] >= 0.5)
		if (pred == truth[int(testy[i])][j]): colist.append(j)
	#print(colist)
	if (len(colist) < sz): continue
#	if (all_perturbed_sensors[0] != 0): continue
#	all_perturbed_sensors = all_perturbed_sensors[:sz]
	atidx = np.random.permutation(len(colist))[:sz]
	#print(atidx)
	for j in range(atidx.shape[0]):
		all_perturbed_sensors.append(atidx[j])
	
	print(all_perturbed_sensors)
	pos = all_perturbed_sensors[0]
	xflag = (interface[pos] >= 0.5)
	print(interface[pos])
	_, __ = output_prob(pos, xflag, interface)
	print(_ / __)

	#XX = deepcopy(mat[i])
	#typex = (XX[pos] >= 0.5)

	per = np.zeros(N)
	sampled_lambda = []
	max_prob = 10000
	min_prob = 0
	dfs3(0)
	print(min_prob, max_prob)
	break
	#el = (mn_prob - min_prob) / mn_prob
	#eu = (max_prob - mx_prob) / mx_prob
	#print("%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t" % (mn_prob, mx_prob, min_prob, max_prob, el, eu))
	XX = deepcopy(mat[i])
	if (min_prob >= 0.5 or max_prob < 0.5):
		B += 1
		#if (xflag == True):
		if (min_prob >= 0.5): A += 1
		#else:
		#	if (max_prob < 0.5): A += 1
	if (xflag == True and trans(XX[pos], eps) >= 0.5):
	#	print(XX[pos])
		w_B += 1
		w_A += 1
	if (xflag == False and trans(XX[pos], -eps) < 0.5):
	#	print(XX[pos])
		w_B += 1
		w_A += 1
	xi, yi = 0, 0
	
	#print(mat[i][pos])
	XX = deepcopy(mat[i])
	xi = 2. * min_prob - 1
	if (xflag == True):
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
