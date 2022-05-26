from fastNLP import Instance, DataSet, Vocabulary, Const
import pickle
import re
import random
from pytorch_transformers import BertTokenizer,BasicTokenizer
from datetime import datetime

model="bert-base-uncased"
name="percent" # in ["stock","percent","price","nasdaq","expectation"]
data=pickle.load(open("{}.pkl".format(name),"rb"))

tkr2name = {"BIDU":['baidu','bidu'], "AAPL":['apple','aapl'], "MSFT":['microsoft','msft'], "INTC":['intel','intc'], "QCOM":['qualcomn','qcom'], "AMD":['amd'], "AMZN":['amazon','amzn'], "NVDA":['nvidia','nvda'], "GOOG":['google','googl','goog']}
tkunify={"BIDU":'[unused1]', "AAPL":'Apple', "MSFT":'Microsoft', "INTC":'Intel', "QCOM":'[unused2]', "AMD":'[unused3]', "AMZN":'Amazon', "NVDA":'[unused4]', "GOOG":'Google'}
never_split=["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"]+["[unused{}]".format(i) for i in range(10)]
tokenizer=BertTokenizer.from_pretrained(model,never_split=never_split)

labeled=data #[x for x in data if x[-1]!=-1]
#random.shuffle(labeled)
n=len(labeled)
train_data=[x for x in labeled if datetime.strptime(x[2],"%m%d%Y").strftime("%Y%m%d")<="20120101"]
test_data=[x for x in labeled if datetime.strptime(x[2],"%m%d%Y").strftime("%Y%m%d")>"20120101"]

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
        if name in ["price","percent"]:
            idx, tk, date, text, p_lis=x
            seq=handle(text)
            seq=transform(seq,tkr2name[tk],tkunify[tk])
            for i,p in enumerate(p_lis):
                new_seq=seq+" [SEP] "+tkunify[tk].lower()+" [SEP] "+p
                if len(dataset)<10:
                    print(new_seq)
                if i==0: label=1
                else: label=0
                new_seq="[CLS] "+new_seq
                new_seq=tokenizer.encode(new_seq)
                """
                seq=["[CLS]"]+word_tokenize(x["raw_text"])
                seq=tokenizer.convert_tokens_to_ids(seq)
                """
                if len(new_seq)>512:
                    new_seq=new_seq[:512]
                    tot+=1
                if cnt.get(label) is None: cnt[label]=1
                else: cnt[label]+=1
                
                if p.startswith("$"): price=p[1:].strip()
                else: price=p.strip()
                
                ins=Instance(idx=idx,tk=tk,date=date,seq=new_seq,label=label,seq_len=len(seq),price=float(price))
                dataset.append(ins)      
            continue
             
        if name in ["stock","expectation","nasdaq"]:
            idx, tk, date, text, label=x
            seq=handle(text)
        if name in ["stock"]:
            seq=transform(seq,tkr2name[tk],tkunify[tk])
            seq=tkunify[tk].lower()+" "+seq
        
        seq="[CLS] "+seq
        seq=tokenizer.encode(seq)
        """
        seq=["[CLS]"]+word_tokenize(x["raw_text"])
        seq=tokenizer.convert_tokens_to_ids(seq)
        """
        if len(seq)>512:
            seq=seq[:512]
            tot+=1
        if label==-1: label=0
        if cnt.get(label) is None: cnt[label]=1
        else: cnt[label]+=1
        ins=Instance(idx=idx,tk=tk,date=date,seq=seq,label=label,seq_len=len(seq))
        dataset.append(ins)       
    
    dataset.set_input("seq","seq_len")
    dataset.set_target("label")
    print(dataset[5])
    print("number:",len(dataset),tot,cnt)
    print()
    return dataset

out={}    
out["train_set"] = make_dataset(train_data)
out["test_set"] = make_dataset(test_data)

with open("{}_dataset.pkl".format(name),"wb") as outfile:
    pickle.dump(out,outfile)
    
