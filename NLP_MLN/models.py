from pytorch_transformers import BertConfig,BertModel
from pytorch_transformers.modeling_utils import SequenceSummary
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def seq_len_to_mask(seq_len,max_len=None):
    if isinstance(seq_len, np.ndarray):
        assert len(np.shape(seq_len)) == 1, f"seq_len can only have one dimension, got {len(np.shape(seq_len))}."
        if max_len is None:
            max_len = int(seq_len.max())
        broad_cast_seq_len = np.tile(np.arange(max_len), (len(seq_len), 1))
        mask = broad_cast_seq_len < seq_len.reshape(-1, 1)
    
    elif isinstance(seq_len, torch.Tensor):
        assert seq_len.dim() == 1, f"seq_len can only have one dimension, got {seq_len.dim() == 1}."
        batch_size = seq_len.size(0)
        if max_len is None:
            max_len = seq_len.max().long()
        broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(seq_len)
        mask = broad_cast_seq_len.lt(seq_len.unsqueeze(1))
    else:
        raise TypeError("Only support 1-d numpy.ndarray or 1-d torch.Tensor.")
    
    return mask   
          
class BertC(nn.Module):
    def __init__(self, name='bert-base-uncased',dropout=0.1,num_class=2):
        super(BertC, self).__init__()
        config = BertConfig.from_pretrained(name)     
        self.bert = BertModel(config) 
        self.proj = nn.Linear(config.hidden_size,num_class)
        self.loss_f=nn.CrossEntropyLoss()    
        self.drop=nn.Dropout(p=dropout)

    def forward(self, src, seq_len, gold=None):
        src_mask=seq_len_to_mask(seq_len,src.size(1))
        out = self.bert(src,attention_mask=src_mask)
        embed=out[1]
        #print(embed.size())
        logits=self.proj(self.drop(embed))
        ret={"pred":logits}
        if gold is not None:
            ret["loss"]=self.loss_f(logits,gold)
        return ret
        
class Bert1(nn.Module):
    def __init__(self, name='bert-base-uncased',dropout=0.1):
        super(Bert1, self).__init__()
        config = BertConfig.from_pretrained(name)     
        self.bert = BertModel(config) 
        self.proj = nn.Linear(config.hidden_size,1)
        self.loss_f=nn.BCEWithLogitsLoss()    
        self.drop=nn.Dropout(p=dropout)

    def forward(self, src, seq_len, gold=None):
        src_mask=seq_len_to_mask(seq_len,src.size(1))
        out = self.bert(src,attention_mask=src_mask)
        embed=out[1]
        #print(embed.size())
        logits=self.proj(self.drop(embed))
        ret={"pred":logits}
        if gold is not None:
            ret["loss"]=self.loss_f(logits,gold.unsqueeze(1).float())
        return ret
 
class PreEmbeddings(nn.Module):
    def __init__(self, init_embedding=None, word_size=None,d_model=None,freeze=True):
        super(PreEmbeddings, self).__init__()
        if init_embedding is None:
            self.embed=nn.Embedding(word_size,d_model)
            self.d_model = d_model
        else:
            self.embed=nn.Embedding.from_pretrained(torch.FloatTensor(init_embedding),freeze=freeze)
            self.d_model = init_embedding.shape[1]

    def forward(self, x):
        return self.embed(x)
        
class PRNN(nn.Module): 
    def __init__(self, src_embed, d_model=256, dropout=0.2,layers=2):
        super(PRNN, self).__init__()
        self.lstm = nn.LSTM(input_size = src_embed.d_model,
                            hidden_size = d_model,
                            num_layers = layers,
                            dropout = dropout,
                            batch_first=True,
                            bidirectional = True)
        self.src_embed = src_embed
        self.proj = nn.Linear(d_model*2, 1)
        self.loss_f=nn.BCEWithLogitsLoss()    
        self.drop=nn.Dropout(p=dropout)

    def forward(self, src, seq_len, gold=None):
        src_mask=seq_len_to_mask(seq_len,src.size(1))
        batch_size,leng=src_mask.size()
        out, (h,c) =self.lstm(self.src_embed(src))
        feat1=self.drop(h.permute(1,0,2).contiguous().view(batch_size,-1))

        feat2=out+((1-src_mask.float())*(-1e9)).unsqueeze(-1).expand_as(out)
        feat2,__=torch.max(feat2.permute(0,2,1),dim=-1)
        feat2=self.drop(feat2)

        logits=self.proj(feat2)         
        ret={"pred":logits}
        if gold is not None:
            ret["loss"]=self.loss_f(logits,gold.unsqueeze(1).float())
        return ret
  
 