import argparse
import pickle
import collections
import logging
import math
import os,sys,time
import random
import numpy as np
import torch
import torch.nn as nn
import fastNLP
from fastNLP import BucketSampler,SequentialSampler,RandomSampler
from fastNLP import Instance, DataSet, DataSetIter,Vocabulary
import models
import utils
from sys import maxsize
# ===-----------------------------------------------------------------------===
# Argument parsing
# ===-----------------------------------------------------------------------===
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, dest="dataset", help=".pkl file to use") 
parser.add_argument("--model", default="bert-base-uncased",dest="model", help="model")         
parser.add_argument("--devi", default="0", dest="devi", help="gpu")
parser.add_argument("--lr", default=1e-5, dest="lr", type=float,help="lr")
parser.add_argument("--clip", default=0, dest="clip", type=float,help="clip")
parser.add_argument("--num-epochs", default=10, dest="num_epochs", type=int,
                    help="Number of step")          
parser.add_argument("--batch-size", default=32, dest="batch_size", type=int,
                    help="Minibatch size of training set")
parser.add_argument("--dropout", default=0.2, dest="dropout", type=float,
                    help="Amount of dropout(not keep rate, but drop rate) to apply to embeddings part of graph")
parser.add_argument("--log-dir", default="result", dest="log_dir",
                    help="Directory where to write logs / serialized models")
parser.add_argument("--task-name", default=time.strftime("%Y-%m-%d-%H-%M-%S"), dest="task_name",
                    help="Name for this task, use a comprehensive one")
parser.add_argument("--old-model", dest="old_model", help="Path to old model for incremental training")
parser.add_argument("--python-seed", dest="python_seed", type=int, default=random.randrange(maxsize),
                    help="Random seed of Python and NumPy")
parser.add_argument("--debug", dest="debug", default=False, action="store_true", help="Debug mode")
parser.add_argument("--targeted", dest="targeted", default=False, action="store_true", help="targeted mode")
parser.add_argument("--always-model", dest="always_model", action="store_true",
                    help="Always serialize model after every epoch")
parser.add_argument("--out", dest="out", default="", help="Path to old model for incremental training")
                    
options = parser.parse_args()
task_name = options.task_name
root_dir = "{}/{}".format(options.log_dir, task_name)
utils.make_sure_path_exists(root_dir)

def init_logger():
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    log_formatter = logging.Formatter("%(message)s")
    logger = logging.getLogger()
    file_handler = logging.FileHandler("{0}/info.log".format(root_dir), mode='w')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger


# ===-----------------------------------------------------------------------===
# Set up logging
# ===-----------------------------------------------------------------------===
logger = init_logger()

# ===-----------------------------------------------------------------------===
# Log some stuff about this run
# ===-----------------------------------------------------------------------===
logger.info(' '.join(sys.argv))
logger.info('')
logger.info(options)

random.seed(options.python_seed)
np.random.seed(options.python_seed % (2 ** 32 - 1))
logger.info('Python random seed: {}'.format(options.python_seed))

# ===-----------------------------------------------------------------------===
# Read in dataset
# ===-----------------------------------------------------------------------===
dataset = pickle.load(open(options.dataset, "rb"))
train_set=dataset["train_set"]
dev_set=dataset["test_set"]
test_set=dataset["test_set"]

if options.debug:
    print("DEBUG MODE")
    #options.num_epochs = 2
    options.batch_size=min(options.batch_size,16)
    train_set=train_set[:200]
    options.num_epochs=2
    
# ===-----------------------------------------------------------------------===
# Build model and trainer
# ===-----------------------------------------------------------------------===

best_model_file_name = "{}/model.bin".format(root_dir)
if options.model.lower().startswith("bert"):
    model=models.BertC(name=options.model, dropout=options.dropout,num_class=2)
else:
    print(len(dataset["vocab"]))
    init_embedding=fastNLP.io.embed_loader.EmbedLoader.load_with_vocab(options.model, dataset["vocab"])
    embedding=models.PreEmbeddings(init_embedding,freeze=False)
    model=models.PRNN(embedding)
    for name,p in model.named_parameters():
        if p.dim() > 1 and p.requires_grad==True and name.find("embed")!=1:
            nn.init.xavier_uniform(p)
            
if options.old_model is not None:
    model.load_state_dict(torch.load(options.old_model)) 
devices=[int(x) for x in options.devi]    
model=nn.DataParallel(model,device_ids=devices)
device = torch.device("cuda:{}".format(devices[0]))      
model=model.to(device)  
optimizer=torch.optim.Adam(model.parameters(), lr = options.lr, eps=1e-6,weight_decay=1e-4)

#train_sampler = BucketSampler(batch_size=options.batch_size, seq_len_field_name='seq_len')
train_sampler=RandomSampler()
dev_sampler = SequentialSampler()
test_batch=DataSetIter(dataset=test_set,batch_size=options.batch_size,sampler=dev_sampler)
    
def to_list(tensor):
    return tensor.detach().cpu().tolist()
    
def tester(test_batch):
    model.eval()
    total = 0.0
    acc = 0.0
    tar_succ=0
    ori=0
    exists_ori=False
    success=[]
    cur_id=0
    with torch.no_grad():
        for batch_x, batch_y in test_batch:
            if len(batch_x["seq_len"])<len(devices):
                continue      
            seq=batch_x["seq"]
            out = model(seq,batch_x["seq_len"])                  
            logits=out["pred"].detach().cpu()
            if logits.size(1)==1:
                pred=torch.sigmoid(logits).squeeze(1).numpy()
                acc+=np.sum((pred>0.5)==(batch_y["label"].numpy()==1))
            else:
                pred=logits.argmax(dim=-1)
                acc += pred.eq(batch_y["label"]).sum().item()
            num=logits.size(0)
            total += num
            
            
    logger.info("ACC: {} {} {}".format(acc/total,acc,total))        
    return acc
    
# start training        
if options.num_epochs>0:                 
    if options.clip!=0:
        torch.nn.utils.clip_grad_norm_(clf.model.parameters(), options.clip)
        
    logger.info("Number training instances: {}".format(len(train_set)))
    logger.info("Number dev instances: {}".format(len(dev_set)))
    logger.info("Number test instances: {}".format(len(test_set)))
    train_batch=DataSetIter(batch_size=options.batch_size, dataset=train_set, sampler=train_sampler)
    dev_batch=DataSetIter(batch_size=options.batch_size, dataset=dev_set, sampler=dev_sampler)
    
    best=0.
    for epoch in range(int(options.num_epochs)):
        logger.info("Epoch {} out of {}".format(epoch + 1, options.num_epochs))
        t1=time.time()
        model.train()
        train_loss=0
        tot=0
        for batch_x,batch_y in train_batch:
            model.zero_grad()
            out=model(batch_x["seq"],batch_x["seq_len"],batch_y["label"])
            loss = torch.mean(out["loss"])
            train_loss += loss.item() 
            tot+=1
            loss.backward()
            optimizer.step()
        
        t2=time.time()      
        train_loss = train_loss / tot
        logger.info("time: {} loss: {}".format(t2-t1,train_loss))
        
        model.eval()
        p = tester(dev_batch)        
        if p > best:
            best = p
            logger.info("- new best score!")
            # Serialize model
            logger.info("Saving model to {}".format(best_model_file_name))
            torch.save(model.module.state_dict(),best_model_file_name)
            #tester(test_batch)    
            
        elif options.always_model:
            logger.info("Saving model to {}".format(best_model_file_name))
            torch.save(model.module.state_dict(),best_model_file_name)

else:
    tester(test_batch)
    
