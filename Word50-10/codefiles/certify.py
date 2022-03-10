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
import numpy as np
import pickle
import warnings
from decimal import *
from scipy.stats import norm

#warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description='Sensing Reasoning Pipeline')
parser.add_argument('--sigma', type=float)
parser.add_argument('--r', type=float)
parser.add_argument('--wH', default = 1, type=float)
args = parser.parse_args()

def p2w(j, prob): # w_G = log(p/(1-p))
	prob = np.clip(prob, 1e-16, 1 - 1e-16) # save for some precision overfloat
	if j < 10:
		return np.log(prob) - np.log(1. - prob)
	else:
		return np.log(prob) - np.log(1. - prob) + args.wH

def trans(pA, delta): # pA => \Phi^{-1}(\Phi(pA) - r / sigma)
	return norm.cdf(norm.ppf(pA) - delta)

# with open("inter.pt", "rb") as tf: # Data loader
# 	(testX, testy) = pickle.load(tf)
test_dataset = torch.load('../data/word10.pt')
testy = test_dataset['test_all_label']

with open("../ckpts/pA/word10_top2_sigma%.2f.pkl" % (args.sigma), "rb") as tf: # pA loader
	## the pA obtained by randomized smoothing is recorded in mat
	## while the index here records the index of the top2 returned from each classifier
	## and we will build the rule based on this index
	mat, index = pickle.load(tf)

with open("word10_rules.pkl", "rb") as tf: # Possible worlds
	## original full rules with 10 + 26 * 5 = 140 dims
	full_rules = pickle.load(tf)


def recursion(cur, J1, J2, S1, S2):
	global min_prob
	global max_prob
	if (cur == len(all_perturbed_sensors)):
		### if there is some overflow, just replace the term inside the np.exp with Decimal(...) ###
		old_z1, z2 = np.exp(world_weight_a[wpos] + J1),  S1 * np.exp(J1)#np.sum(np.exp(world_weight_a + J1))
		z1, old_z2 =  np.exp(world_weight_b[wpos] + J2), S2 * np.exp(J2)#np.sum(np.exp(world_weight_b + J2))
		min_prob = max(min_prob, old_z1 / old_z2)
		if old_z1/old_z2 > min_prob:
			best_per = per.copy()
		max_prob = min(max_prob, z1 / z2)
		return ;
	
	curpos = all_perturbed_sensors[cur]
	per[curpos] = 0
	sa, sb = 0, 0
	for i in range(len(related_worlds[curpos])):
		wid = related_worlds[curpos][i]
		sa += np.exp(world_weight_a[wid]) * (np.exp(interface_L[curpos]) - 1)
		sb += np.exp(world_weight_b[wid]) * (np.exp(interface_U[curpos]) - 1)
		world_weight_a[wid] += interface_L[curpos]
		world_weight_b[wid] += interface_U[curpos]
		
	recursion(cur + 1, J1, J2, S1 + sa, S2 + sb)

	sa, sb = 0, 0
	for i in range(len(related_worlds[curpos])):
		wid = related_worlds[curpos][i]
		sa += np.exp(world_weight_a[wid] - interface_L[curpos]) * (np.exp(interface_U[curpos] - 1))
		sb += np.exp(world_weight_b[wid] - interface_U[curpos]) * (np.exp(interface_L[curpos] - 1))
		world_weight_a[wid] += interface_U[curpos] - interface_L[curpos]
		world_weight_b[wid] += interface_L[curpos] - interface_U[curpos]
	
	per[curpos] = -1
	recursion(cur + 1, J1 - (interface_U[curpos] - interface[curpos]), 
			J2 - (interface_L[curpos] - interface[curpos]), S1 + sa, S2 + sb)

	for i in range(len(related_worlds[curpos])):
		wid = related_worlds[curpos][i]
		world_weight_a[wid] -= interface_U[curpos]
		world_weight_b[wid] -= interface_L[curpos]
	
def output_prob(pos, _interface):
	weight_of_all_worlds, weight_of_worlds_contains_me = 0, 0
	for i in range(len(rules)):
		current_world_weight = 0
		for j in range(len(rules[i])):
			if (rules[i][j]): current_world_weight += p2w(j,_interface[j])
		current_world_weight = np.exp(current_world_weight)
		weight_of_all_worlds += current_world_weight
		if (rules[i][pos] == True): weight_of_worlds_contains_me += current_world_weight
	return weight_of_worlds_contains_me, weight_of_all_worlds


# print(rules)
eps = args.r / args.sigma 

global all_perturbed_sensors
global sampled_lambda

#all_perturbed_sensors = []
#for i in range(12): 
#	all_perturbed_sensors.append(i)
#for i in range(7): all_perturbed_sensors.append(i + 12)
#all_perturbed_sensors.extend([11 + 12, 12 + 12, 13 + 12])


all_cases = 0
for i in range(mat.shape[0]): all_cases += (mat[i][testy[i].item()] >= 0.5)


tp, A, w_A = 0, 0, 0
for i in range(mat.shape[0]):

    index = pick[i].astype(bool)
    ## build new rule ###
    rules = full_rules[:,index]
    cur_mat = mat[i][index]
    pos = testy[i].item()
    N = rules.shape[1]

    wpos = []
    for j in range(len(rules)):
        if (rules[j][pos] == True): wpos = j
    min_prob, max_prob = 0, 1
    all_perturbed_sensors = [t for t in range(N)] # all sensors are perturbed
    related_worlds = [[] for t in range(N)]
    for t in range(len(rules)):
        for j in range(len(rules[t])):
            if (rules[t][j]): related_worlds[j].append(t)

    interface, interface_U, interface_L = np.zeros(N), np.zeros(N), np.zeros(N)
    world_weight_a, world_weight_b = np.zeros(rules.shape[0]), np.zeros(rules.shape[0])


    for j in range(N): 
        interface[j] = cur_mat[j]
        if (interface[j] >= 0.5):
            interface_U[j] = p2w(j, interface[j])
            interface_L[j] = p2w(j, trans(interface[j], eps))
        else:
            interface_U[j] = p2w(j, trans(interface[j], -eps))
            interface_L[j] = p2w(j, interface[j])

    for j in range(N):
        if (j in all_perturbed_sensors): continue
        for k in range(len(related_worlds[j])):
            wid = related_worlds[j][k]
            world_weight_a[wid] += p2w(j, interface[j])
            world_weight_b[wid] += p2w(j, interface[j])

    #if (interface[pos] < 0.5): continue # Ignore originally failed cases ====> can be evaluated later
    xflag = (interface[pos] >= 0.5)
    _, __ = output_prob(pos, interface)
    print("Interface confidence: %.5f" % (interface[pos]))
    print("Marginal confidence before perturbation: %.5f" % (_ / __))

    per = np.zeros(N)
    sampled_lambda = []
    recursion(0, 0, 0, np.sum(np.exp(world_weight_a)), np.sum(np.exp(world_weight_b)))

    without_knowledge = [trans(interface[pos], eps), trans(interface[pos], -eps)]
    print("[Theoretical bound w/o knowledge] Min: %.5f, Max: %.5f" % (without_knowledge[0], without_knowledge[1]))
    print("[Theoretical bound w/ knowledge] Min: %.5f, Max: %.5f" % (min_prob, max_prob))

    if (without_knowledge[0] >= 0.5): A += 1
    if (min_prob >= 0.5): w_A += 1
#     if (without_knowledge[0] >= 0.5 and xflag == True): A += 1
#     if (without_knowledge[1] <= 0.5 and xflag == False): A += 1

#     if (min_prob >= 0.5 and xflag == True): w_A += 1
#     if (max_prob <= 0.5 and xflag == False): w_A += 1

    tp += 1
    if (tp % 1 == 0):
        print("[%d/%d] w/o knowledge: %.4f, w/ knowledge: %.4f" % (tp, mat.shape[0], 100 * A / tp, 100 * w_A / tp))
    print("w/o knowledge: %.4f, w/ knowledge: %.4f" % (100 * A / tp, 100 * w_A / tp))
