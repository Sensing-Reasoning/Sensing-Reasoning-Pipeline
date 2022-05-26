from fastNLP import Instance, DataSet, Vocabulary, Const
import pickle
import re
import random
from pytorch_transformers import BasicTokenizer
from datetime import datetime

model="bert-base-uncased"
name="stock" # in ["stock","nasdaq"]
data=pickle.load(open("{}.pkl".format(name),"rb"))

tkr2name = {"BIDU":['baidu','bidu'], "AAPL":['apple','aapl'], "MSFT":['microsoft','msft'], "INTC":['intel','intc'], "QCOM":['qualcomn','qcom'], "AMD":['amd'], "AMZN":['amazon','amzn'], "NVDA":['nvidia','nvda'], "GOOG":['google','googl','goog']}
tkunify={"BIDU":'[unused1]', "AAPL":'Apple', "MSFT":'Microsoft', "INTC":'Intel', "QCOM":'[unused2]', "AMD":'[unused3]', "AMZN":'Amazon', "NVDA":'[unused4]', "GOOG":'Google'}
never_split=["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"]+["[unused{}]".format(i) for i in range(10)]
tokenizer=BasicTokenizer(do_lower_case=True,never_split=never_split)

labeled=data #[x for x in data if x[-1]!=-1]
#random.shuffle(labeled)
n=len(labeled)
train_data=[x for x in labeled if datetime.strptime(x[2],"%m%d%Y").strftime("%Y%m%d")<="20120101"]
test_data=[x for x in labeled if datetime.strptime(x[2],"%m%d%Y").strftime("%Y%m%d")>"20120101"]

all_set=DataSet()

def handle(text):
    sent=text.strip()
    sent=sent.replace("``","\"")
    sent=sent.replace("''","\"")
    sent=re.sub(r'\( [A-Z]{4}\.[A-Z] \)',"",sent)        
    return sent
    
def transform(content,lis,tk):
    seq=content.split(" ")
    res=[tk if x.lower() in lis else x for x in seq]
    return " ".join(res)
        
def make_dataset(data):
    dataset=DataSet()
    tot=0
    cnt={}
    for x in data:                   
        if name in ["stock","expectation","nasdaq"]:
            idx, tk, date, text, label=x
            seq=handle(text)
        if name in ["stock"]:
            seq=transform(seq,tkr2name[tk],tkunify[tk])
            seq=tkunify[tk].lower()+" "+seq
        
        seq=tokenizer.tokenize(seq)
        if cnt.get(label) is None: cnt[label]=1
        else: cnt[label]+=1
        ins=Instance(idx=idx,tk=tk,date=date,seq=seq,label=label,seq_len=len(seq))
        dataset.append(ins)       
        all_set.append(ins)
        
    dataset.set_input("seq","seq_len")
    dataset.set_target("label")
    print(dataset[5]["seq"])
    print("number:",len(dataset),tot,cnt)
    print()
    return dataset

out={}    
out["train_set"] = make_dataset(train_data)
out["test_set"] = make_dataset(test_data)

vocab=Vocabulary(padding='[PAD]', unknown='[UNK]').from_dataset(all_set,field_name='seq')
vocab.index_dataset(out["train_set"], field_name='seq',new_field_name='seq')
vocab.index_dataset(out["test_set"], field_name='seq',new_field_name='seq')
out["vocab"]=vocab
with open("lstm_{}.pkl".format(name),"wb") as outfile:
    pickle.dump(out,outfile)
    
