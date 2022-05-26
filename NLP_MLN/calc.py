import pickle
import math
import numpy as np
import random
from copy import deepcopy
import time

data=pickle.load(open("mln_big_set.pkl","rb"))
print(len(data))

epsilons=[0.1,0.2,0.3,0.5,0.7,0.8,0.9]
attack_lis=[0,0,0,0]
plot_name="attack all groups"
global w_arith1,w_arith2
w_arith1=10
w_arith2=10

eps_bounds={}
for eps in epsilons:
    eps_bounds[eps]=pickle.load(open("{}_{}_{}.pkl".format("".join([str(x) for x in attack_lis]),eps,w_arith1),"rb"))    

def mln(x,y,h,w1,w2,w3,w4,flag=1):
    marginal1=[0 for i in range(len(x["prices"]))]
    marginal2=[0 for i in range(len(y["prices"]))]
    marginal3=[0,0]
    marginal5=[0,0]
    marginal4=[0 for i in range(len(h["prices"]))]
    s=0
    global w_arith1,w_arith2    
    for i in range(len(marginal1)):
        for j in range(len(marginal2)):
            for k in range(2):
                for l in range(len(marginal4)):
                    weight = w1[i] + w2[j]+w3[k]+w4[l]
                    ff=1 if y["prices"][j]>x["prices"][i] else -1
                    if abs(y["prices"][j]-(1+ff*0.01*h["prices"][l])*x["prices"][i])<0.2:
                        weight = weight + w_arith2
                    if k==int(y["prices"][j]>x["prices"][i]):
                        weight = weight + w_arith1
                    marginal1[i] = marginal1[i] + math.exp(weight)
                    marginal2[j] = marginal2[j] + math.exp(weight)
                    marginal3[k] = marginal3[k] + math.exp(weight)
                    marginal4[l] = marginal4[l] + math.exp(weight)
                    
                    s = s + math.exp(weight)
    if flag==1:                        
        for i in range(len(marginal1)):
            marginal1[i] = marginal1[i] / s
        for i in range(len(marginal2)):
            marginal2[i] = marginal2[i] / s
        for i in range(len(marginal4)):
            marginal4[i] = marginal4[i] / s
        for i in range(2):
            marginal3[i] = marginal3[i] / s
        res=[]
        res.extend(marginal1)
        res.extend(marginal2)
        res.extend(marginal3)
        res.extend(marginal4)
    elif flag==2:
        res=[]
        res.extend(marginal1)
        res.extend(marginal2)
        res.extend(marginal3)
        res.extend(marginal4)
    elif flag==0:
        res=[s]
    return marginal1,marginal2,marginal3,marginal4, res

tot=0  
result=[[{"certify":0,"certify_true":0,"certify_k":0,"certify_true_k":0} for j in range(4)] for i in range(len(epsilons))]
def check(lis,itv):
    for x,y in lis:
        if y>itv[0]: return 0
    return 1

h2=[[] for i in range(len(epsilons))]
margin_plot=[[[],[]] for i in range(len(epsilons))]

for idx, ins in enumerate(data):
    x,y,z,h=ins
    p1,p2,p3,p4=x["prob"],y["prob"],z["prob"],h["prob"]
    p1,p2,p3,p4=np.clip(p1,1e-5,1-1e-5),np.clip(p2,1e-5,1-1e-5),np.clip(p3,1e-5,1-1e-5),np.clip(p4,1e-5,1-1e-5)
    if not(np.argmax(p1)==x["label"] and np.argmax(p2)==y["label"] and np.argmax(p3)==z["label"] and np.argmax(p4)==h["label"]):
        continue

    p_lis=[p1,p2,p3,p4]
    loc_lis=[x["label"],len(p1)+y["label"],len(p1)+len(p2)+z["label"],len(p1)+len(p2)+len(p3)+h["label"]]
    label_lis=[x["label"],y["label"],z["label"],h["label"]]
    """
    ff=True
    for i,ty in enumerate(attack_lis):
        if ty<3 and np.argmax(p_lis[i])!=label_lis[i]:
            ff=False
            break
    if ff==False: continue
    """
    w1=np.log(p1/(1-p1)).tolist()
    w2=np.log(p2/(1-p2)).tolist()
    w3=np.log(p3/(1-p3)).tolist()
    w4=np.log(p4/(1-p4)).tolist()
    marginal1,marginal2,marginal3,marginal4, res=mln(x,y,h,w1,w2,w3,w4)
    care=marginal2[label_lis[1]]
    tot+=1

    top1=y["label"]
    top2=np.array(p2).argsort()[::-1][1]
    
    for k,eps in enumerate(epsilons):
        w1=np.log(p1/(1-p1)).tolist()
        w2=np.log(p2/(1-p2)).tolist()
        w3=np.log(p3/(1-p3)).tolist()
        w4=np.log(p4/(1-p4)).tolist()
        w_lis=[w1,w2,w3,w4]
        for loc, ty in enumerate(attack_lis):
            label=label_lis[loc]
            new_p_lis=[]
            for i,p in enumerate(p_lis[loc]):
                new_p=p
                pa,pb=p,p
                if i==label and ty in [0,2]: 
                    new_p=max(1e-5,p-eps)
                    pa=new_p
                w_lis[loc][i]=math.log(new_p/(1-new_p))
                new_p_lis.append((pa,pb))
            fcer=check([x for i,x in enumerate(new_p_lis) if i!=label],new_p_lis[label])
            result[k][loc]["certify"]+=fcer
            result[k][loc]["certify_true"]+=fcer&(np.argmax(w_lis[loc])==label)
            
        margin_plot[k][0].append(p2[top1]-eps-p2[top2])
                                 
        bounds=eps_bounds[eps]
        w1,w2,w3,w4=w_lis
        marginal1,marginal2,marginal3,marginal4, res=mln(x,y,h,w1,w2,w3,w4)
        m_lis=[marginal1,marginal2,marginal3,marginal4]
        for loc,mar in enumerate(m_lis):
            label=label_lis[loc]
            mx=max([i for i in range(len(mar))],key=lambda x : bounds[idx][loc][x][1])
            fcer=check([x for i,x in enumerate(bounds[idx][loc]) if i!=mx],bounds[idx][loc][label])
            
            result[k][loc]["certify_k"]+=fcer
            result[k][loc]["certify_true_k"]+=fcer&(mx==label)
        
        h2[k].append(bounds[idx][1][label_lis[1]]+(care,))

        margin_plot[k][1].append(bounds[idx][1][top1][0]-min(1,bounds[idx][1][top2][1]))
"""
for k,eps in enumerate(epsilons):
    a,b=np.histogram(margin_plot[k][0],bins=8,range=(-1,1))
    aa,bb=np.histogram(margin_plot[k][1],bins=8,range=(-1,1))
    n=len(margin_plot[k][0])
    with open("bins_{}_{}.txt".format(plot_name,eps),"w") as f:
        for i in range(len(a)):
            f.write("[{},{}]\t{}\t{}\n".format(b[i],b[i+1],aa[i]/n,a[i]/n))

import matplotlib.pyplot as plt
for k,eps in enumerate(epsilons):
    fig = plt.figure()
    plt.style.use('seaborn-whitegrid')
    #ax1 = fig.add_axes((0.1, 0.2, 0.8, 0.7))
    #ax1.set_title("{} eps = {}".format(plot_name,eps))
    #print(len(margin_plot[k][0]))
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.title("{} eps = {}".format(plot_name,eps))
    plt.xlabel("Margin of w/o knowledge")
    plt.ylabel("Margin of w knowledge")
    plt.plot([-1,1],[-1,1],color='r')
    plt.scatter(margin_plot[k][0],margin_plot[k][1])
    plt.savefig('{}_{}.png'.format(plot_name,eps))

    fig = plt.figure()
    plt.style.use('seaborn-whitegrid')
    plt.xlim(-1,1)
    plt.hist(margin_plot[k][0],bins=100,range=(-1,1),label="w/o knowledge")
    plt.hist(margin_plot[k][1],bins=100,range=(-1,1),label="w knowledge")
    plt.title("{} eps = {}".format(plot_name,eps))
    plt.legend(loc='upper right',fontsize=12);
    plt.savefig('hist_{}_{}.png'.format(plot_name,eps))
    with open("{}_{}.txt".format(plot_name,eps),"w") as f:
        for i in range(len(margin_plot[k][0])):
            f.write("{} {}\n".format(margin_plot[k][1][i],margin_plot[k][0][i]))
            
exit()
""" 

for k,eps in enumerate(epsilons):
    with open("h2_{}_{}_{}.pkl".format("".join([str(x) for x in attack_lis]),eps,w_arith1),"wb") as f:
        pickle.dump(h2[k],f)
     
print(tot)
for k, res in enumerate(result):
    print(epsilons[k])
    print(res)
    for loc in [1]:
        A=1.*res[loc]["certify_true"]/tot
        B=1.*res[loc]["certify_true"]/res[loc]["certify"]
        C=1.*res[loc]["certify"]/tot
        A_k=1.*res[loc]["certify_true_k"]/tot
        B_k=1.*res[loc]["certify_true_k"]/res[loc]["certify_k"]
        C_k=1.*res[loc]["certify_k"]/tot
        print(A,B,C,A_k,B_k,C_k)
        
        
    
