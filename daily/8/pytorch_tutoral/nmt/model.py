import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import numpy as np
from utils import subseqenceMask,standardMask 
from metrics.evaluation_utils import blue

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(Q,K,V,mask=None,dropout=None):
    '''
    Q:(...,T1,dk)
    K:(...,T2,dk)
    V:(...,T2,dv)
    mask:(...,T1,T2)
    using compoents in V to represent(encode) Q

    return:
        out=(...,T1,dv)
        p_attn=(...,T1,T2)
    '''
    d=Q.size(-1)

    S=torch.matmul(Q,K.transpose(-2,-1))/(d**0.5)
    if mask is not None:
        S.masked_fill_(mask==0,-1e9)
    #S.data[mask==0]=1e-9
    p_attn=F.softmax(S,dim=-1)
    if dropout:
        p_attn=dropout(p_attn)
    out=torch.matmul(p_attn,V)

    return out,p_attn
class MultiAttentionLayer(nn.Module):
    def __init__(self,d,h,drop=0.1):
        '''
        d: hidden size
        h:split factor
        '''
        super().__init__()
        assert d%h==0,'d must divide by h'
        self.dk=d//h
        self.h=h
        self.d=d

        self.n1=nn.Linear(d,d)
        self.n2=nn.Linear(d,d)
        self.n3=nn.Linear(d,d)
        self.n4=nn.Linear(d,d)
        self.dropout=nn.Dropout(drop) 

    def forward(self,Q,K,V,mask=None):
        '''
            Q:(N,T1,d)
            K:(N,T2,d)
            V:(N,T2,d)
            mask:(N,T1,T2)

            return
            
            out:(N,T1,d)
            p_attn:(N,T1,T2)
        '''
        N,_,d=Q.size()
        #(N,h,T1,dk)
        q =self.n1(Q).view(N,-1,self.h,self.dk).transpose(-2,-3)
        k=self.n2(K).view(N,-1,self.h,self.dk).transpose(-2,-3)
        v=self.n3(V).view(N,-1,self.h,self.dk).transpose(-2,-3)
        #(N,1,T1,T2)
        if mask is not None:
            mask=mask.unsqueeze(1)
        #(N,h,T1,dk)
        out,self.attn=attention(q,k,v,mask,self.dropout)
        out=self.n4(out.transpose(-2,-3).contiguous().view(N,-1,d))
        return out
class FFN(nn.Module):
    def __init__(self,d,h,dropout=0.1):
        '''
        d:model size
        h:hidden size

        '''
        super().__init__()
        self.fn1=nn.Linear(d,h)
        self.fn2=nn.Linear(h,d)
        self.dropout=nn.Dropout(dropout) 
        self.d=d
    def forward(self,x):
        x=self.fn2(self.dropout(F.relu(self.fn1(x))))
        return x
class NormLayer(nn.Module):
    def __init__(self,size,eps=1e-6):
        '''
        size:feature size,this is last dim of your input
        '''
        super().__init__()
        self.eps=eps
        self.a=nn.Parameter(torch.ones(size))
        self.b=nn.Parameter(torch.zeros(size))

    def forward(self,x):
        mean=torch.mean(x,-1,keepdim=True)
        std=torch.std(x,-1,keepdim=True)
        return (x-mean)/(std+self.eps)*self.a+self.b
class SkipLayer(nn.Module):
    def __init__(self,d,drop=0.1):
        super().__init__()
        self.norm=NormLayer(d)
        self.dropout=nn.Dropout(drop)
    def forward(self,x,layer):
        return x+self.dropout(layer(self.norm(x)))
class EncoderLayer(nn.Module):
    def __init__(self,attn,ffn,drop):
        '''
        attn:MultiAttentionLayer
        ffn:feed forward Layer
        drop:drop factor for skip connection
        '''
        super().__init__()
        self.attn=attn
        self.ffn=ffn
        self.skiplayers=clones(SkipLayer(ffn.d,drop),2)
        self.d=ffn.d
    def forward(self,x,mask):
        '''
        x:(N,T,D)
        mask:(N,1,T)
        
        return:(N,T,D)
        '''
        x=self.skiplayers[0](x,lambda x:self.attn(x,x,x,mask))
        x=self.skiplayers[1](x,self.ffn)

        return x
class DecoderLayer(nn.Module):
    def __init__(self,attn1,attn2,ffn,drop):
        super().__init__()
        self.attn1=attn1
        self.attn2=attn2
        self.ffn=ffn
        self.skiplayers=clones(SkipLayer(ffn.d,drop),3)
        self.d=ffn.d 
    def forward(self,x,memory,mask,memory_mask):
        '''
        x:previous output (N,T,D)
        memory:encoder output(N,TA,D)
        mask:x's mask (N,T,T)
        menory_mask:(N,1,TA)
        '''
        x=self.skiplayers[0](x,lambda x:self.attn1(x,x,x,mask))
        x=self.skiplayers[1](x,lambda x:self.attn2(x,memory,memory,memory_mask))
        x=self.skiplayers[2](x,self.ffn)

        return x
class Encoder(nn.Module):
    def __init__(self,block,N):
        '''
        block:a EncoderLayer obj
        N:replace block N times
        '''
        super().__init__()
        self.model=clones(block,N)
        self.norm=NormLayer(block.d)
    def forward(self,x,mask):
        for m in self.model:
            x=m(x,mask)
        return self.norm(x)
class Decoder(nn.Module):
    def __init__(self,block,N):
        '''
        block:a DecoderLayer obj
        N:replace block N times
        '''
        super().__init__()
        self.model=clones(block,N)
        self.norm=NormLayer(block.d)
    def forward(self,x,memory,mask,memory_mask):
        for m in self.model:
            x=m(x,memory,mask,memory_mask)
        return self.norm(x)
class EmbedingLayer(nn.Module):
    def __init__(self,V,d):
        '''
        V:vocab num
        d:model size
        '''
        super().__init__()
        self.model=nn.Embedding(V,d)
        self.d=d
        self.V=V
    def forward(self,x):
        return self.model(x)*(self.d**0.5)
class PositionEncoding(nn.Module):
    def __init__(self,Tmax,d,dropout):
        '''
        Tmax:max sequence length
        d:model size 
        dropout:drop factor after this layer's output
        '''
        super().__init__()
        self.dropout=nn.Dropout(dropout)
        pe=torch.zeros(Tmax,d)

        trange=torch.arange(Tmax).unsqueeze(1).float()
        drange=torch.arange(0,d,2).float()

        divTerm=torch.exp(-drange*(math.log(10000)/d))
        pe[0:Tmax,0:d:2]=torch.sin(trange*divTerm)
        pe[0:Tmax,1:d:2]=torch.cos(trange*divTerm)
        self.d=d
        self.Tmax=Tmax
        self.register_buffer('pe',pe)
    def forward(self,x):
        T=x.size(-2)
        return self.dropout(x+self.pe[:T])

class Generator(nn.Module):
    def __init__(self,V,d):
        '''
            V:target size
            d:model size
        '''
        super().__init__()
        self.proj=nn.Linear(d,V)
        self.d=d
        self.V=V
    def forward(self,x):
        return F.log_softmax(self.proj(x),dim=-1)

class EncoderDecoder(nn.Module):
    def __init__(self,src_emb,tgt_emb,encoder,decoder,proj):
        super().__init__()

        self.src_emb=src_emb
        self.tgt_emb=tgt_emb
        self.encoder=encoder
        self.decoder=decoder
        self.proj=proj
    def encode(self,x,xmask):
        return self.encoder(self.src_emb(x),xmask)
    def decode(self,x,memory,xmask,memory_mask):
        return self.decoder(self.tgt_emb(x),memory,xmask,memory_mask)
    def forward(self,x,y,xmask,ymask):
        memory=self.encoder(self.src_emb(x),xmask)
        out=self.decoder(self.tgt_emb(y),memory,ymask,xmask)
        return self.proj(out)

def makeModel(Vsrc,Vtgt,Tmax=100,d=512,dropout=0.1):
    '''
    Vsrc:source vocab size
    Vtgt:target vocab size
    Tmax:max sequence length
    d:model size
    drop:drop factor
    '''
    ######prepare base element of transformer model
    attn=MultiAttentionLayer(d,h=8,drop=dropout)
    ffn=FFN(d,2048,dropout)
    c=copy.deepcopy

    ######prepare for encoder and decoder
    encoder_blk=EncoderLayer(c(attn),c(ffn),dropout)
    decoder_blk=DecoderLayer(c(attn),c(attn),c(ffn),dropout)
    encoder=Encoder(encoder_blk,6)
    decoder=Decoder(decoder_blk,6)

    ####prepare for embedding layer

    src_emb=nn.Sequential(EmbedingLayer(Vsrc,d),PositionEncoding(Tmax,d,dropout))
    tgt_emb=nn.Sequential(EmbedingLayer(Vtgt,d),PositionEncoding(Tmax,d,dropout))

    ####prepare for generator
    generator=Generator(Vtgt,d)

    model=EncoderDecoder(src_emb,tgt_emb,encoder,decoder,generator)

    cn=0
    for p in model.parameters():
        cn+=np.prod(p.size())
        if p.dim()>1:
            nn.init.xavier_uniform_(p)
    print('Total # of trainable %d'%(cn))
    return model
class LabelSmoothingLoss(nn.Module):
    def __init__(self,size,paddingidx,smooth=0):
        '''
        size:vocab size
        paddingidx:if target ==paddingidx,the loss for that target is 0
        smooth:how much to smooth the label
        '''
        super().__init__()
        self.smooth=smooth
        self.criterion=nn.KLDivLoss(size_average=False)
        self.confidence=1-smooth 
        self.size=size
        self.paddingidx=paddingidx

    def forward(self,x,y,normalizer): 
        '''
        x:(N,size):
        y:(N,):sparse encoding
        
        this network first create smooth version one-hot representation
        Y',if some element of y equal padding idx,coresponding one-hot Y
        will all be zero. Then compute KL(x,Y)

        return: sum(KL(x,Y)) ,sum up all element of KL term(no batch average!)
        '''
        N=x.size(0)


        true_dist=x.data.clone()
        true_dist.fill_(self.smooth/(self.size-2))

        #true_dist[torch.arange(N),y]=self.confidence
        true_dist.scatter_(1,y.unsqueeze(1),self.confidence)
        true_dist[:,self.paddingidx]=0
        
        mask=torch.nonzero(y==self.paddingidx) #(K,1)
        if mask.dim()>1:
            mask=mask.squeeze()
            #true_dist[mask]=0
            true_dist.index_fill_(0,mask,0)
        return self.criterion(x,true_dist.requires_grad_())/normalizer

class ComputeLoss():
    def __init__(self,generator,optimizer):
        '''
        generator:a softLable Layer that to conmute loss
        optimizer: optimize loss create by generator
        '''
        self.generator=generator
        self.optimizer=optimizer

    def __call__(self,x,y,normalizer):
        '''
            x:(N,T,V)
            y:(N,T)
            normalizer:

            return:average loss 
        '''
        V=x.size(-1)
        loss=self.generator(x.contiguous().view(-1,V),y.contiguous().view(-1),normalizer)
        loss.backward()
        if self.optimizer:
            with torch.no_grad():
                self.optimizer.step()
                self.optimizer.zero_grad()
        return loss.item()
def greedyDecoder(x,xmask,model,maxlen=100,startidx=1,unk=0):
    '''
    x:(N,T)
    mask:(N,T,T) or (N,T,T)
    '''
    N=x.size(0)
    y=torch.zeros(N,1).to(x.device).long().fill_(startidx)
    memory=model.encode(x,xmask)
    for i in range(maxlen):
        ymask=subseqenceMask(y)&standardMask(y,unk)
        out=model.proj(model.decode(y,memory,ymask,xmask))  #(N,T,V)
        out=torch.argmax(out,-1)[:,-1:]   #(N,T)
        y=torch.cat((y,out.long()),-1)
    return y[:,1:].cpu().numpy()

def run_train_epoch(data_iter,model,loss_func,epoch,display=10):
    model.train()
    for i,batch in enumerate(data_iter):
        out=model(batch.x,batch.yin,batch.xmask,batch.ymask)
        loss=loss_func(out,batch.yout,batch.ntoken)
        if i %display==0:
            print('Epoch %d,step:%d,loss:%.4f'%(epoch,i,loss))
    return {
            'model':model.state_dict(),
            'epoch':epoch,
            'loss':loss
            }
    
def run_eval(data_iter,model,decoder=None):
    ref=[]
    my=[]
    model.eval()
    if decoder==None:
        decoder=lambda x,xmask,m:greedyDecoder(x,xmask,m,maxlen=100,startidx=data_iter.ds.sos_idx,unk=data_iter.ds.padding_idx)
    for i,batch in enumerate(data_iter):
        y=decoder(batch.x,batch.xmask,model)
        
        _ref=data_iter.translate_tgt(batch.yout)
        _my=data_iter.translate_tgt(y)
        print(_ref[0])
        print('-----------------------')
        print(_my[0])
        print('-----------------------')
        ref.extend(_ref)
        my.extend(_my)
    score=blue(my,ref)
    print('Blue score:{}'.format(score))
    return score
if __name__=='__main__':
    pass
