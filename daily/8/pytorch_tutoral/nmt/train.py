import torch
import torch.optim as optim
from model import makeModel


class CustomOptimizer():
    def __init__(self,dmodel,factor,warmup,optimizer):
        self.d=dmodel**(-0.5)
        self.factor=factor 
        self.warmup=warmup
        self._step=0
        self.optimizer=optimizer
        
    def rate(self):
        return min(self._step**(-0.5),self._step*(self.warmup**(-1.5)))*self.d

    def step(self):
        self._step+=1
        for p in self.optimizer.param_groups:
            p['lr']=self.rate()
        print(self.lr*self.rate())
        self.optimizer.step()
    def zero_grad(self):
        self.optimizer.zero_grad()
def get_std_opt(model):
    return CustomOptimizer(model.src_emb[0].d,2,4000,optim.Adam(model.parameters(),lr=0.0,betas=(0.9,0.98),eps=1e-9))
if __name__=='__main__':
    model=makeModel(100,200)
    get_std_opt(model)
