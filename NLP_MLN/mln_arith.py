import models
from fastNLP import Instance, DataSet, Vocabulary, Const
import pickle
import torch
import math
import torch.nn.functional as F
import numpy as np
from fastNLP import BucketSampler,SequentialSampler,RandomSampler
from fastNLP import Instance, DataSet, DataSetIter
from tqdm import tqdm
import datetime as dt
from datetime import datetime

#expect=pickle.load(open("expectation_dataset.pkl","rb"))
#nasdaq=pickle.load(open("nasdaq_dataset.pkl","rb"))
price=pickle.load(open("price_dataset.pkl","rb"))
stock=pickle.load(open("stock_dataset.pkl","rb"))

device = torch.device("cuda:{}".format(2))
price_model=models.BertC("bert-base-uncased", dropout=0,num_class=2)
stock_model=models.BertC("bert-base-uncased", dropout=0,num_class=2)

price_model.load_state_dict(torch.load("result/price/model.bin"))
stock_model.load_state_dict(torch.load("result/stock/model.bin"))

price_model=price_model.to(device)
stock_model=stock_model.to(device)

price_model.eval()
stock_model.eval()
#expect_model=models.BertC("bert-base-uncased", dropout=0,num_class=2)
#nasdaq_model=models.BertC("bert-base-uncased", dropout=0,num_class=2)

#expect_model.load_state_dict(torch.load("result/expect1/model.bin"))
#nasdaq_model.load_state_dict(torch.load("result/nasdaq/model.bin"))
#expect_model=expect_model.to(device)
#nasdaq_model=nasdaq_model.to(device)

#expect_model.eval()
#nasdaq_model.eval()

percent=pickle.load(open("percent_dataset.pkl","rb"))
percent_model=models.BertC("bert-base-uncased", dropout=0,num_class=2)
percent_model.load_state_dict(torch.load("result/percent1/model.bin"))
percent_model=percent_model.to(device)
percent_model.eval()

epsilons=[0.1,0.2,0.3,0.4,0.5]
res=[{"td_acc":0,"td_k_acc":0,"tm_acc":0,"tm_k_acc":0,"s_acc":0,"s_k_acc":0,"p_acc":0,"p_k_acc":0,"all":0,"all_k":0} for i in range(len(epsilons)+1)]

def mln(x,y,h,w1,w2,w3,w4):
    marginal1=[0 for i in range(len(x["prices"]))]
    marginal2=[0 for i in range(len(y["prices"]))]
    marginal3=[0,0]
    marginal5=[0,0]
    marginal4=[0 for i in range(len(h["prices"]))]
    s=0
    w_arith1=0
    w_arith2=0
    
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
                            
    for i in range(len(marginal1)):
        marginal1[i] = marginal1[i] / s
    for i in range(len(marginal2)):
        marginal2[i] = marginal2[i] / s
    for i in range(len(marginal4)):
        marginal4[i] = marginal4[i] / s
    for i in range(2):
        marginal3[i] = marginal3[i] / s
        
    return marginal1,marginal2,marginal3,marginal4
    
def get_probability(test_set,model):
    if isinstance(test_set,list):
        test_set=DataSet(test_set)
        test_set.set_input("idx","seq","seq_len")
        test_set.set_target("label")
    else:
        test_set.set_input("idx")
    model.eval()
    p_lis=[]
    batch_size=128
    test_batch=DataSetIter(dataset=test_set,batch_size=batch_size,sampler=SequentialSampler())
    cur=0
    with torch.no_grad():
        for batch_x, batch_y in test_batch:
            seq,seq_len=batch_x["seq"].to(device),batch_x["seq_len"].to(device)
            out = model(seq,seq_len)
            if out["pred"].size(1)==1:
                ps=torch.sigmoid(out["pred"])
            else: ps=F.softmax(out["pred"],dim=-1)
            ps=ps.detach().cpu().numpy()
            for i,p in enumerate(ps):
                x=test_set[cur]
                assert x["idx"]==int(batch_x["idx"][i])
                if len(p)==1:
                    dic={"prob":np.array([float(1-p[0]),float(p[0])]),"label":x["label"],"idx":x["idx"],"tk":x["tk"],"date":x["date"]}
                else:
                    dic={"prob":p,"label":x["label"],"idx":x["idx"],"tk":x["tk"],"date":x["date"]}
                if "price" in x.fields:
                    dic["price"]=x["price"]
                p_lis.append(dic)
                cur+=1
            
    return p_lis
    
def set_key(lis):
    dic={}
    for x in lis:
        if dic.get((x["tk"],x["date"])) is None:
            dic[(x["tk"],x["date"])]=[x]
        else:
            dic[(x["tk"],x["date"])].append(x)
    return dic
    
def rearrange(lis):
    dic={}
    for x in lis:
        if dic.get(x["idx"]) is None:
            dic[x["idx"]]={"prob":[float(x["prob"][1])],"idx":x["idx"],"tk":x["tk"],"date":x["date"],"prices":[x["price"]]}
        else:
            dic[x["idx"]]["prob"].append(float(x["prob"][1]))
            dic[x["idx"]]["prices"].append(x["price"])
        if x["label"]==1:
            dic[x["idx"]]["label"]=len(dic[x["idx"]]["prob"])-1
            
    nw=[]
    acc=0
    for x in dic.keys():
        y=dic[x]
        acc+=int(np.argmax(y["prob"])==y["label"])
        y["prob"]=np.array(y["prob"])
        nw.append(y)
    print(acc,1.0*acc/len(nw),len(nw))
    print(nw[:3])
    return nw
   
def set_key_date(lis):
    dic={}
    for x in lis:
        if dic.get(x["date"]) is None:
            dic[x["date"]]=[x]
        else:
            dic[x["date"]].append(x)
    return dic
    
price_p=get_probability(price["test_set"],price_model)
price_p=rearrange(price_p)

#expect_p=get_probability(expect["test_set"],expect_model)
#nasdaq_p=get_probability(nasdaq["test_set"],nasdaq_model)
stock_p=get_probability(stock["test_set"],stock_model)

#dic_expect=set_key(expect_p)   
#dic_nasdaq=set_key_date(nasdaq_p)
dic_price=set_key(price_p)
dic_stock=set_key(stock_p)   

percent_p=get_probability(percent["test_set"],percent_model)
percent_p=rearrange(percent_p)
dic_percent=set_key(percent_p)   

tot=0
def tomorrow(date):
    date=datetime.strptime(date,"%m%d%Y")
    ndate=(date+dt.timedelta(days=1)).strftime("%m%d%Y")
    return ndate
    
cnt=0
all_set=[]
for ky in tqdm(dic_price.keys()):
    tm_ky=(ky[0],tomorrow(ky[1])) 
    if dic_price.get(tm_ky) is None: continue
    if dic_stock.get(tm_ky) is None: continue
    #if dic_nasdaq.get(tm_ky[-1]) is None: continue
    #if dic_expect.get(tm_ky) is None: continue
    if dic_percent.get(tm_ky) is None: continue
    for x in dic_price[ky]:
        if len(x["prices"])==1: continue
        ff=0
        for y in dic_price[tm_ky]: 
            if len(y["prices"])==1: continue
            for z in dic_stock[tm_ky]:
                if int(y["prices"][y["label"]]>x["prices"][x["label"]])!=z["label"]: continue
                for h in dic_percent[tm_ky]:
                    if len(h["prices"])==1: continue
                    f=1 if y["prices"][y["label"]]>x["prices"][x["label"]] else -1
                    if abs(y["prices"][y["label"]]-x["prices"][x["label"]]*(1+0.01*f*h["prices"][h["label"]]))>0.2: continue
                    all_set.append((x,y,z,h))
                    tot+=1
                    p1,p2,p3,p4=x["prob"],y["prob"],z["prob"],h["prob"]
                    p1,p2,p3,p4=np.clip(p1,1e-5,1-1e-5),np.clip(p2,1e-5,1-1e-5),np.clip(p3,1e-5,1-1e-5),np.clip(p4,1e-5,1-1e-5)
                    w1=np.log(p1/(1-p1)).tolist()
                    w2=np.log(p2/(1-p2)).tolist()
                    w3=np.log(p3/(1-p3)).tolist()
                    w4=np.log(p4/(1-p4)).tolist()
                    
                    res[0]["td_acc"]+=int(np.argmax(p1)==x["label"])
                    res[0]["tm_acc"]+=int(np.argmax(p2)==y["label"])
                    res[0]["s_acc"]+=int(np.argmax(p3)==z["label"])
                    res[0]["p_acc"]+=int(np.argmax(p4)==h["label"])
                    res[0]["all"]+=int(np.argmax(p1)==x["label"] and np.argmax(p2)==y["label"] and np.argmax(p3)==z["label"] and np.argmax(p4)==h["label"])
                    marginal1,marginal2,marginal3,marginal4=mln(x,y,h,w1,w2,w3,w4)
                        
                    res[0]["td_k_acc"]+=int(np.argmax(marginal1)==x["label"])
                    res[0]["tm_k_acc"]+=int(np.argmax(marginal2)==y["label"])
                    res[0]["s_k_acc"]+=int(np.argmax(marginal3)==z["label"])
                    res[0]["p_k_acc"]+=int(np.argmax(marginal4)==h["label"])
                    res[0]["all_k"]+=int(np.argmax(marginal1)==x["label"] and np.argmax(marginal2)==y["label"] and np.argmax(marginal3)==z["label"] and np.argmax(marginal4)==h["label"])
                    
                    #attack stock
                    for k,epsilon in enumerate(epsilons):  
                        
                        for i,p in enumerate(p1):
                            new_p=p
                            if i==x["label"]: new_p=max(1e-5,p-epsilon)
                            elif (i==x["label"]-1) or (x["label"]==0 and i==x["label"]+1): new_p=min(1-1e-5,p+epsilon)
                            w1[i]=math.log(new_p/(1-new_p))
                        
                        for i,p in enumerate(p2):
                            new_p=p
                            if i==y["label"]: new_p=max(1e-5,p-epsilon)
                            elif (i==y["label"]-1) or (y["label"]==0 and i==y["label"]+1): new_p=min(1-1e-5,p+epsilon)
                            w2[i]=math.log(new_p/(1-new_p))
                        """
                        for i,p in enumerate(p3):
                            new_p=p
                            if i==1: new_p=max(1e-5,p-epsilon)
                            if i==0: new_p=min(1-1e-5,p+epsilon)
                            
                            w3[i]=math.log(new_p/(1-new_p))
                          
                        for i,p in enumerate(p4):
                            new_p=p
                            if i==h["label"]: new_p=max(1e-5,p-epsilon)
                            elif (i==h["label"]-1) or (h["label"]==0 and i==h["label"]+1): new_p=min(1-1e-5,p+epsilon)
                            w4[i]=math.log(new_p/(1-new_p))
                        """  
                        res[k+1]["td_acc"]+=int(np.argmax(w1)==x["label"])
                        res[k+1]["tm_acc"]+=int(np.argmax(w2)==y["label"])
                        res[k+1]["s_acc"]+=int(np.argmax(w3)==z["label"])
                        res[k+1]["p_acc"]+=int(np.argmax(w4)==h["label"])
                        res[k+1]["all"]+=int(np.argmax(w1)==x["label"] and np.argmax(w2)==y["label"] and np.argmax(w3)==z["label"] and np.argmax(w4)==h["label"])
                        
                        marginal1,marginal2,marginal3,marginal4=mln(x,y,h,w1,w2,w3,w4)
                            
                        res[k+1]["td_k_acc"]+=int(np.argmax(marginal1)==x["label"])
                        res[k+1]["tm_k_acc"]+=int(np.argmax(marginal2)==y["label"])
                        res[k+1]["s_k_acc"]+=int(np.argmax(marginal3)==z["label"])
                        res[k+1]["p_k_acc"]+=int(np.argmax(marginal4)==h["label"])
                        res[k+1]["all_k"]+=int(np.argmax(marginal1)==x["label"] and np.argmax(marginal2)==y["label"] and np.argmax(marginal3)==z["label"] and np.argmax(marginal4)==h["label"])                
                        """
                        if epsilon==0.4 and int(np.argmax(marginal1)==x["label"])<int(np.argmax(w1)==x["label"]):
                            print(x["label"],x["prices"])
                            print(y["label"],y["prices"])
                            print(z["label"])
                            print(p1,p2,p3)
                            print(w1,w2,w3)
                            print(marginal1)
                            print(marginal2)
                            print(marginal3)
                            exit()
                        """
        cnt+=ff
        
print(tot,cnt)
for x in res:
    """
    for p in x.keys():
        x[p]=x[p]*100.0/tot
    """
    print(x)

print(len(all_set))
print(all_set[0])
with open("mln_big_set.pkl","wb") as f:
    pickle.dump(all_set,f)    


