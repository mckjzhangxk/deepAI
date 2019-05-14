import torch
import torch.nn as nn



def MyKL(x,y,reduction='mean'):
    '''
    x:(...,V):log prob
    x:(...,V):distribution
    '''
    
    L=y*(torch.log(y+1e-9)-x) #(...,V)

    if reduction=='mean':
        return torch.mean(L)
    if reduction=='sum':
        return torch.sum(L)
    if reduction=='batchmean':
        L=torch.sum(L,-1)
        return torch.mean(L)
if __name__=='__main__':
    reduce='sum'

    N,V=128,32
    X=nn.LogSoftmax(-1)(torch.randn(N,V))
    Y=nn.Softmax(-1)(torch.randn(N,V))

    l1=MyKL(X,Y,reduce)
    print('My ans:',l1)
    '''
    size_average=False,reduce=True, == reduction='sum'
    size_average=True,reduce=True, == reduction='mean' 
    
    '''
    l2=nn.KLDivLoss(size_average=False)(X,Y)
    print('torch ans:',l2)
