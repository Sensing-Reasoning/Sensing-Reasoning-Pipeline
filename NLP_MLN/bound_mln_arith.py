import pickle
import math
import numpy as np
import random
from copy import deepcopy
import time
import csv
import sys
from tqdm import tqdm

data=pickle.load(open("mln_big_set.pkl","rb"))
print(len(data))
global lmb_sample_lis,attack_lis,C,ins

results=[[],[],[],[]]
cnt=[0,0,0,0]
C=0.5
if len(sys.argv)>1:
    C=float(sys.argv[1])
    
attack_lis=[2,2,3,3]
print(attack_lis)
#lmb_sample_lis=np.arange(-5,5,0.25)
lmb_sample_lis=[None,None]
lmb_sample_lis[0]=np.arange(-2.001,-0.0001,0.25).tolist()+np.arange(1.001,2.002,0.25).tolist()
lmb_sample_lis[1]=np.arange(-2.001,-1.0001,0.25).tolist()+np.arange(0.001,1.001,0.05).tolist()

def mln(x,y,h,w1,w2,w3,w4,flag=1):
    marginal1=[0 for i in range(len(x["prices"]))]
    marginal2=[0 for i in range(len(y["prices"]))]
    marginal3=[0,0]
    marginal5=[0,0]
    marginal4=[0 for i in range(len(h["prices"]))]
    s=0
    
    global w_arith1,w_arith2
    w_arith1=10
    w_arith2=10
    
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

def max_list(a,b):
    if a is None : return b
    if b is None : return a
    res=[]
    for x,y in zip(a,b):
        res.append(max(x,y))
    return res

def min_list(a,b):
    if a is None : return b
    if b is None : return a
    res=[]
    for x,y in zip(a,b):
        res.append(min(x,y))
    return res

def l_list(a,b):
    for x,y in zip(a,b):
        if x>y: return False
    return True

def calc(lmb_lis,eps_lis,flag):
    out=None
    x,y,z,h=ins
    p1,p2,p3,p4=x["prob"],y["prob"],z["prob"],h["prob"]
    p1,p2,p3,p4=np.clip(p1,1e-5,1-1e-5),np.clip(p2,1e-5,1-1e-5),np.clip(p3,1e-5,1-1e-5),np.clip(p4,1e-5,1-1e-5)
    w1=np.log(p1/(1-p1)).tolist()
    w2=np.log(p2/(1-p2)).tolist()
    w3=np.log(p3/(1-p3)).tolist()
    w4=np.log(p4/(1-p4)).tolist()
    ww1=deepcopy(w1)
    ww2=deepcopy(w2)
    ww3=deepcopy(w3)
    ww4=deepcopy(w4)
    w_lis=[ww1,ww2,ww3,ww4]
    old_w_lis=[w1,w2,w3,w4]
    p_lis=[p1,p2,p3,p4]
    label_lis=[x["label"],y["label"],z["label"],h["label"]]
    diff=0

    for loc, ty in enumerate(attack_lis):
        label=label_lis[loc]
        for i,p in enumerate(p_lis[loc]):
            new_p=p
            if i==label and ty in [0,2]:
                new_p=max(1e-5,p-eps_lis[loc][0])
                w_lis[loc][i]=math.log(new_p/(1-new_p))
                diff+=lmb_lis[loc][0]*(old_w_lis[loc][i]-w_lis[loc][i])
            elif ty in [1,2] and ((i==label-1) or (label==0 and i==label+1)):
                new_p=min(1-1e-5,p+eps_lis[loc][1])
                w_lis[loc][i]=math.log(new_p/(1-new_p))
                diff+=lmb_lis[loc][1]*(w_lis[loc][i]-old_w_lis[loc][i])
            w_lis[loc][i]=math.log(new_p/(1-new_p))
       
    ww1,ww2,ww3,ww4=w_lis
    marginal1,marginal2,marginal3,marginal4, res=mln(x,y,h,ww1,ww2,ww3,ww4,flag)
    diff=np.array([diff])
    res=np.array(res)*np.exp(diff)

    return res.tolist()

def dfs2(dep,eps_choice,func,flag):
    global eps_lis,ans
    if dep==4:
        ans=func(ans,calc(lmb_lis,eps_lis,flag))
    else:
        if attack_lis[dep]==2:
            for eps1 in eps_choice[dep][0]:
                for eps2 in eps_choice[dep][1]:
                    eps_lis[dep][0]=eps1
                    eps_lis[dep][1]=eps2
                    dfs2(dep+1,eps_choice,func,flag)
        elif attack_lis[dep] in [0,1]:
            for eps in eps_choice[dep][attack_lis[dep]]:
                eps_lis[dep][attack_lis[dep]]=eps
                dfs2(dep+1,eps_choice,func,flag)
        else: dfs2(dep+1,eps_choice,func,flag)
            
def dfs(dep):
    global th_mi_prob, th_mx_prob,lmb_lis,ans,eps_lis
    if dep==4:
        eps1_choice=[[None,None] for i in range(len(lmb_lis))]
        eps2_choice=[[None,None] for i in range(len(lmb_lis))]
        for i in range(len(lmb_lis)):
            lmb=lmb_lis[i][0]
            if lmb is not None:
                z1,z2=[],[]
                if lmb>0 and lmb<1:
                    z1.append(C)
                    z1.append(0)

                    z2.append(C)
                    z2.append(0)
                    assert False
                elif lmb>=1:
                    z1.append(C)
                    z2.append(0)
                elif lmb<=0:
                    z1.append(0)
                    z2.append(C)
                eps1_choice[i][0]=deepcopy(z1)
                eps2_choice[i][0]=deepcopy(z2)
            lmb=lmb_lis[i][1]    
            if lmb is not None:
                z1,z2=[],[]
                if lmb>-1 and lmb<0:
                    z1.append(C)
                    z1.append(0)

                    z2.append(C)
                    z2.append(0)
                    assert False
                elif lmb>=0:
                    z1.append(C)
                    z2.append(0)
                elif lmb<=-1:
                    z1.append(0)
                    z2.append(C)
                eps1_choice[i][1]=deepcopy(z1)
                eps2_choice[i][1]=deepcopy(z2)
        eps_lis=[[None,None] for i in range(4)]
        ans=None
        dfs2(0,eps1_choice,max_list,2)
        z1max=np.array(ans)
        
        eps_lis=[[None,None] for i in range(4)]
        ans=None
        dfs2(0,eps2_choice,min_list,0)
        z2min=np.array(ans)
        
        th_mx_prob=min_list(th_mx_prob,(z1max/z2min).tolist())
        
        eps_lis=[[None,None] for i in range(4)]
        ans=None
        dfs2(0,eps2_choice,min_list,2)
        z1min=np.array(ans)
        
        eps_lis=[[None,None] for i in range(4)]
        ans=None
        dfs2(0,eps1_choice,max_list,0)
        z2max=np.array(ans)
        
        th_mi_prob=max_list(th_mi_prob,(z1min/z2max).tolist())

    else:
        cur=attack_lis[dep]
        if cur==2:
            for lmb1 in lmb_sample_lis[0]:
                for lmb2 in lmb_sample_lis[1]:
                    lmb_lis[dep][0]=lmb1
                    lmb_lis[dep][1]=lmb2
                    dfs(dep+1)
        elif cur in [0,1]:
            for lmb1 in lmb_sample_lis[cur]:
                lmb_lis[dep][cur]=lmb1
                dfs(dep+1)
        else:
            dfs(dep+1)
   
def split(a,lis):
    res=[]
    for x,y in zip(lis[:-1],lis[1:]):
        res.append(a[x:y])
    return res

bounds={}
for idx,ins in tqdm(enumerate(data)):
    t1=time.time()
    x,y,z,h=ins
    p1,p2,p3,p4=x["prob"],y["prob"],z["prob"],h["prob"]
    p1,p2,p3,p4=np.clip(p1,1e-5,1-1e-5),np.clip(p2,1e-5,1-1e-5),np.clip(p3,1e-5,1-1e-5),np.clip(p4,1e-5,1-1e-5)
    if not(np.argmax(p2)==y["label"]):
        continue
    w1=np.log(p1/(1-p1)).tolist()
    w2=np.log(p2/(1-p2)).tolist()
    w3=np.log(p3/(1-p3)).tolist()
    w4=np.log(p4/(1-p4)).tolist()
    p_lis=[p1,p2,p3,p4]
    loc_lis=[x["label"],len(p1)+y["label"],len(p1)+len(p2)+z["label"],len(p1)+len(p2)+len(p3)+h["label"]]
    label_lis=[x["label"],y["label"],z["label"],h["label"]] 
    em_mi_prob=None
    em_mx_prob=None
    
    # empirical
    for epoch in range(100):    
        epsilons=[random.random()*C for i in range(8)]
        w_lis=[w1,w2,w3,w4]
        for loc, ty in enumerate(attack_lis):
            label=label_lis[loc]
            for i,p in enumerate(p_lis[loc]):
                new_p=p
                if ty==2:
                    if i==label: new_p=max(1e-5,p-epsilons[loc*2])
                    elif (i==label-1) or (label==0 and i==label+1): new_p=min(1-1e-5,p+epsilons[loc*2+1])
                elif ty==1:
                    if (i==label-1) or (label==0 and i==label+1): new_p=min(1-1e-5,p+epsilons[loc*2+1])
                elif ty==0:
                    if i==label: new_p=max(1e-5,p-epsilons[loc*2])                  
                w_lis[loc][i]=math.log(new_p/(1-new_p))
                                           
        w1,w2,w3,w4=w_lis
        
        marginal1,marginal2,marginal3,marginal4, res=mln(x,y,h,w1,w2,w3,w4)
        
        em_mi_prob=min_list(em_mi_prob,res)
        em_mx_prob=max_list(em_mx_prob,res)
    
    # Theoretical
    
    th_mi_prob=None
    th_mx_prob=None
    lmb_lis=[[None,None] for i in range(4)]
    dfs(0)

    em_mi,th_mi=np.array(em_mi_prob),np.array(th_mi_prob)
    em_mx,th_mx=np.array(em_mx_prob),np.array(th_mx_prob)
    rl=(em_mi-th_mi)/em_mi
    ru=(th_mx-em_mx)/em_mx

    """
    print(th_mi_prob)
    print(th_mx_prob)
    print(loc_lis,time.time()-t1)
    """
   
    lis=[0,len(p1),len(p1)+len(p2),len(p1)+len(p2)+len(p3),len(p4)+len(p1)+len(p2)+len(p3)]
    combine=[(x,y) for x,y in zip(th_mi_prob,th_mx_prob)]
    tt=split(combine,lis)
    bounds[idx]=tt
    
    for loc,ty in enumerate(attack_lis):
        if ty not in [0,1,2]: continue   
        el,eu=em_mi[loc_lis[loc]],em_mx[loc_lis[loc]]
        tl,tu=th_mi[loc_lis[loc]],th_mx[loc_lis[loc]]
        rl=(el-tl)/el
        ru=(tu-eu)/eu
        if tu-tl<=C:
            cnt[loc]+=1
        results[loc].append({"el":el,"eu":eu,"tl":tl,"tu":tu,"rl":rl,"ru":ru})
    
print(cnt)
for loc,result in enumerate(results):
    if len(result)==0: continue
    with open('{}_{}_{}_{}.csv'.format("".join([str(x) for x in attack_lis]),C,loc+1,w_arith1), 'w', newline='',encoding="utf8") as csvfile:
         writer = csv.DictWriter(csvfile, fieldnames=["el","eu","tl","tu","rl","ru"])
         writer.writeheader()
         for x in result:
             writer.writerow(x)        

with open("{}_{}_{}.pkl".format("".join([str(x) for x in attack_lis]),C,w_arith1),"wb") as f:
    pickle.dump(bounds,f)
