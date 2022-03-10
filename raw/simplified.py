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
#warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--sigma', type=float)
parser.add_argument('--r', type=float)
args = parser.parse_args()

N = 25 # [0, 12) => main sensor bool var, [12, 25) => attr sensor bool var

def p2w(prob): # w_G = log(p/(1-p))
	prob = np.clip(prob, 1e-20, 1 - 1e-20) # save for some precision overfloat
	return np.log(prob) - np.log(1. - prob)

def trans(pA, delta): # pA => \Phi^{-1}(\Phi(pA) - r / sigma)
	return norm.cdf(norm.ppf(pA) - delta)

with open("inter.pt", "rb") as tf: # Data loader
	(testX, testy) = pickle.load(tf)

with open("pA_%.2f.pkl" % (args.sigma), "rb") as tf: # pA loader
	mat = pickle.load(tf)



def recursion(cur):
	global min_prob
	global max_prob
	if (cur == len(all_perturbed_sensors)):
		return ;
		print(sampled_lambda)
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
		#old, new = "", ""
		#for i in range(N):
		#	old += "%.5f\t" % (interface[i])
		#	new += "%.5f\t" % (_interface[i])
		#print(old)
		#print(new)
		J = 0
		for i in range(len(all_perturbed_sensors)):
			cur_lambda = sampled_lambda[i]
			cur_pos = all_perturbed_sensors[i]
			#delta = _interface[cur_pos] - interface[cur_pos]
			delta = p2w(_interface[cur_pos]) - p2w(interface[cur_pos])
			J += cur_lambda * delta
		#print("Z1")
		#print(np.exp(J))
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
		#old, new = "", ""
		#for i in range(N):
		#	old += "%.5f\t" % (interface[i])
		#	new += "%.5f\t" % (_interface[i])
		#print(old)
		#print(new)
		J = 0
		for i in range(len(all_perturbed_sensors)):
			cur_lambda = sampled_lambda[i]
			cur_pos = all_perturbed_sensors[i]
			#delta = _interface[cur_pos] - interface[cur_pos]
			delta = p2w(_interface[cur_pos]) - p2w(interface[cur_pos])
			J += cur_lambda * delta
		#print("Z2")

		#print(np.exp(J))
		z2 = output_prob(pos, xflag, _interface)[1] * np.exp(J)	
		min_prob = max(min_prob, z1 / z2)
		#print(z1 / z2)
		##### Upper bound computation #####
		for i in range(N):
			if (per[i] != 0):
				if (interface[i] >= 0.5):
					_interface[i] = min(interface[i], trans(interface[i], -per[i] * eps))
				else:
					_interface[i] = max(interface[i], trans(interface[i], -per[i] * eps))
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
					_interface[i] = min(interface[i], trans(interface[i], per[i] * eps))
				else:
					_interface[i] = max(interface[i], trans(interface[i], per[i] * eps))
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
	
	for l in [-1.00, 0.00]:
		sampled_lambda.append(l)
		if (l >= -1e-8): per[all_perturbed_sensors[cur]] = 1
		else: per[all_perturbed_sensors[cur]] = -1
		recursion(cur + 1)
		sampled_lambda.pop()
	
def output_prob(pos, flag, _interface):
	weight_of_all_worlds, weight_of_worlds_contains_me = 0, 0 #Decimal(0), Decimal(0)
	for i in range(len(rules)):
		current_world_weight = 0
		#print("########### %d ##########" % (i))
		for j in range(len(rules[i])):
			#if (j >= 7 + 12 and j <= 10 + 12): continue
			if (rules[i][j]): 
		#		print("%d\%.2f" % (j, p2w(_interface[j])))
				current_world_weight += p2w(_interface[j])
		current_world_weight = np.exp(current_world_weight)
		weight_of_all_worlds += current_world_weight
		if (rules[i][pos] == True):
			weight_of_worlds_contains_me += current_world_weight
			#print(current_world_weight)
	return weight_of_worlds_contains_me, weight_of_all_worlds

	#if (flag == True): 	else:
	#	#print((weight_of_all_worlds - weight_of_worlds_contains_me)/weight_of_all_worlds)
	#	return weight_of_all_worlds - weight_of_worlds_contains_me, weight_of_all_worlds

with open("newrules.pt", "rb") as tf: # Possible worlds
	rules = pickle.load(tf)

print(rules)
eps = args.r / args.sigma 

global all_perturbed_sensors
global sampled_lambda

all_perturbed_sensors = []
for i in range(12): all_perturbed_sensors.append(i)
#for i in range(7): all_perturbed_sensors.append(i + 12)
#all_perturbed_sensors.extend([11 + 12, 12 + 12, 13 + 12])

#all_perturbed_sensors = [i for i in range(N)] # all sensors are perturbed

all_cases = 0
for i in range(mat.shape[0]): all_cases += (mat[i][testy[i].item()] >= 0.5)

tp, A, w_A = 0, 0, 0
for i in range(mat.shape[0]):
	interface = np.zeros(N)
	for j in range(N): interface[j] = mat[i][j]
	
	pos = testy[i].item() # Certify the #ground-truth dimension >= 0.5
	#if (interface[pos] < 0.5): continue # Ignore originally failed cases ====> can be evaluated later
	xflag = (interface[pos] >= 0.5)
	#print(interface[pos])	
	_, __ = output_prob(pos, True, interface)
	print("Interface confidence: %.5f" % (interface[pos]))
	print("Marginal confidence before perturbation: %.5f" % (_ / __))
	#print(interface)
	#print(pos)
	
	#if (interface[pos] >= 0.5 and _ / __ < 0.5):
	#	print("Interface confidence: %.5f" % (interface[pos]))
	#	print("Marginal confidence before perturbation: %.5f" % (_ / __))
	#	print(i)
	#	s1, s2 = "", ""
	#	for j in range(N):
	#		s1 += "%.5f\t" % (interface[j])
	#		s2 += "%.5f\t" % (p2w(interface[j]))
	#	print(s1)
	#	print(s2)
	
	per = np.zeros(N)
	sampled_lambda = []
	min_prob, max_prob = 0, 1
	recursion(0)

	without_knowledge = [trans(interface[pos], eps), trans(interface[pos], -eps)]
	print("[Theoretical bound w/o knowledge] Min: %.5f, Max: %.5f" % (without_knowledge[0], without_knowledge[1]))
	print("[Theoretical bound w/ knowledge] Min: %.5f, Max: %.5f" % (min_prob, max_prob))

	if (without_knowledge[0] >= 0.5 and xflag == True): A += 1
	if (without_knowledge[1] <= 0.5 and xflag == False): A += 1

	if (min_prob >= 0.5 and xflag == True): w_A += 1
	if (max_prob <= 0.5 and xflag == False): w_A += 1

	tp += 1
	if (tp % 10 == 0):
		print("[%d/%d] w/o knowledge: %.4f, w/ knowledge: %.4f" % (tp, all_cases, 100 * A / tp, 100 * w_A / tp))
print("w/o knowledge: %.4f, w/ knowledge: %.4f" % (100 * A / tp, 100 * w_A / tp))
