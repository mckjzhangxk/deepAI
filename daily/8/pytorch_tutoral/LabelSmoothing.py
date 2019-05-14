import torch
import torch.nn as nn
import numpy as np

class LabelSmoothing(nn.Module):
    def __init__(self,size,paddingidx,smooth=0):
        super().__init__()
        self.smooth=smooth
        self.criterion=nn.KLDivLoss(size_average=False)
        self.confidence=1-smooth 
        self.size=size
        self.paddingidx=paddingidx

    def forward(self,x,y): 
        '''
        x:(N,size):
        y:(N,):sparse encoding

        return:
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
        return self.criterion(x,true_dist)

        
if __name__=='__main__': 
    layer=LabelSmoothing(5,0,0.1)
    x=torch.tensor([[0.4,0.3,0.1,0.1,0.1],
                    [0.3,0.2,0.2,0.1,0.2],
                    [0.2,0.2,0.2,0.2,0.2]])
    y=torch.tensor([2,1,0]).long()
    
    z=layer(x.log(),y)

    print(z)
