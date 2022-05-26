import pickle
from datetime import datetime
import csv
from nltk.tokenize import sent_tokenize
import re

Tickers=["BIDU", "AAPL", "MSFT", "INTC", "QCOM", "AMD", "AMZN", "NVDA", "GOOG"]
tkr2name = {"BIDU":['baidu','bidu'], "AAPL":['apple','aapl'], "MSFT":['microsoft','msft'], "INTC":['intel','intc'], "QCOM":['qualcomn','qcom'], "AMD":['amd'], "AMZN":['amazon','amzn'], "NVDA":['nvidia','nvda'], "GOOG":['google','googl','goog']}

lis=pickle.load(open("Reuters_raw.pkl","rb"))

def get_true_price():
    path="relative_label"
    percent={}
    for tk in Tickers:
        new_path="{}/{}/{}".format(path,tk,"return_vs_nasdaq.csv")
        with open(new_path) as f:
            f_csv = csv.reader(f)
            for i, row in enumerate(f_csv):
                if i==0: continue
                date=datetime.strptime(row[0].strip(),"%m%d%Y").strftime("%m%d%Y")
                percent[(tk,date)]={"relative":float(row[4]),"absolute":float(row[1])}

    path="10202006_11202013_prices"
    price={}
    for tk in Tickers:
        new_path="{}/{}.csv".format(path,tk)
        with open(new_path) as f:
            f_csv = csv.reader(f)
            for i, row in enumerate(f_csv):
                if i==0: continue
                try:
                    date=datetime.strptime(row[0].strip(),"%Y/%m/%d").strftime("%m%d%Y")
                except:
                    date=datetime.strptime(row[0].strip(),"%Y-%m-%d").strftime("%m%d%Y")
                price[(tk,date)]={"open":float(row[1]),"close":float(row[4])}
                
    print(price[("QCOM","07242013")])
    return price, percent

def check(sent,search):
    a=sent.lower()
    for x in search:
        if a.find(x)!=-1:
            return True
    return False

def expectation(sent,prefer):
    s=0
    for pos in prefer:
        s+=sent.count(pos)
    return s

def gen_ins(tk,date,sent):
    global tot,tt,cnt
    lis=re.findall(r'\$[0-9]+\.[0-9]{2}[\.,; ]',sent)
    if len(lis)==0:
        return
    if len(lis)==1:
        true_price=lis[0][1:-1]
        """
        if price.get((tk,date)) is not None:
            gold=float(price[(tk,date)]["close"])
            if abs(float(true_price)-gold)>1:
                return
        """
        if len(re.findall(r'\$[0-9]+\.[0-9]{2} (million|billion)',sent))>0:
            return 
        new_lis=re.findall(r'[$ ][0-9]+\.[0-9]+[\.,; ]',sent)
        new_lis=new_lis+re.findall(r'[$ ][0-9]+[,; ]',sent)
        new_lis=[x[1:-1] for x in new_lis if x[1:-1]!=true_price]
        p_lis=[true_price]+new_lis[:3]
    else:
        if price.get((tk,date)) is None:
            return
        
        gold=float(price[(tk,date)]["close"])
        p_lis=sorted(lis, key=lambda x: abs(float(x[1:-1])-gold))
        p_lis=[x[1:-1] for x in p_lis]
        if abs(float(p_lis[0])-gold)>1:
            """
            print(sent)
            print(tk,date,gold)
            print()
            """
            return
    """
    if len(p_lis)<2:
        print(sent)
        print(p_lis)
        print()
    """
    cnt+=len(p_lis)
    stock.append((tot,tk,date,sent,p_lis))
    tot+=1
    
price,percent=get_true_price()
tot=0
stock=[]
cnt=0
tt=0
for i,(tk,date,title,text) in enumerate(lis):
    sents=sent_tokenize(text)
    for j,sent in enumerate(sents):
        if j==0: context=sent.strip()
        else: context=sents[j-1]+" "+sent.strip()
        if re.search(r'\$[0-9]+\.[0-9]{2}',sent) is not None:
            """
            for p in Tickers:
                if check(sent,tkr2name[p]):
                    gen_ins(p,date,sent)
            continue
            """
            if check(sent,tkr2name[tk]):
                gen_ins(tk,date,sent)
            else:
                ff=False
                for p in Tickers:
                    if check(sent,tkr2name[p]):
                        gen_ins(p,date,sent)
                        ff=True
                        break
  
print(stock[-5:])
print(tot,cnt,tt)
with open("price.pkl","wb") as f:
    pickle.dump(stock,f)
    
    
