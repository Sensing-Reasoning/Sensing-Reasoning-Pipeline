import re
import os
import sys
import utils
import pickle
from tqdm import tqdm

Tickers=["BIDU", "AAPL", "MSFT", "INTC", "QCOM", "AMD", "AMZN", "NVDA", "GOOG"]

def handle_single(path):
    with open(path,"r",encoding="utf-8") as f:
        cnt=0
        title=""
        content=None
        for i,line in enumerate(f.readlines()):
            #print(i,cnt,len(line.strip()))
            if line.startswith("--"):
                cnt+=1
                if cnt==1:
                    title=line.strip()
                    title=title.replace("-- ","")
                elif cnt==4: content=""
                continue
            if len(line.strip())==0: continue                    
            if content is not None:
                line=line.strip()
                if len(content)==0:
                    if line.find(" (Reuters) - ")!=-1:
                        line=line.split(" (Reuters) - ")[1]
                    line=line.lstrip()
                content+=line+" "
           
    if content is None: content=""    
    return title,content

def handle_files(path,date,tk=None):
    cnt=0
    data=[]
    date=utils.unify_date(date)
    for name in os.listdir(path):
        cur=os.path.join(path,name)
        if os.path.isfile(cur):
            cnt+=1
            title,content=handle_single(cur)
            
            if len(title)==0: continue
            elif len(content)==0:
                print(cur)
                continue

            data.append((tk,date,title,content))
                       
    return data,cnt
    
path="Reuters"
s=0
Rdata=[]
for tk in os.listdir(path):
    if tk in Tickers:
        print(tk,end=" ")
        new_path=os.path.join(path,tk)
        for date in os.listdir(new_path):
            if len(date)==8:
                data,cnt=handle_files(os.path.join(new_path,date),date,tk)
                s+=cnt
                Rdata.extend(data)
                
print(len(Rdata),s)
print(Rdata[-4:])
with open("Reuters_raw.pkl","wb") as outfile:
    pickle.dump(Rdata,outfile)

