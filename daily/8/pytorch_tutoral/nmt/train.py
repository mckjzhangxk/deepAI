import torch
import torch.optim as optim
from model import makeModel,LabelSmoothingLoss,ComputeLoss,run_train_epoch,run_eval 
from dataset import MyDataSet,MyDataLoader


class CustomOptimizer():
    def __init__(self,dmodel,factor,warmup,optimizer):
        self.d=dmodel
        self.factor=factor 
        self.warmup=warmup
        self._step=0
        self.optimizer=optimizer
        
    def rate(self):
        #return min(self._step**(-0.5),self._step*(self.warmup**(-1.5)))*(self.d**(-0.5))*self.factor
        return min((self._step*((self.warmup)**(-1.5)))     ,(self._step**(-0.5)))  *(self.d**(-0.5))*self.factor

    def step(self):
        self._step+=1
        for p in self.optimizer.param_groups:
            p['lr']=self.rate()
        self.optimizer.step()
    def zero_grad(self):
        self.optimizer.zero_grad()
def get_std_opt(model):
    return CustomOptimizer(model.src_emb[0].d,2,100,optim.Adam(model.parameters(),lr=0.0,betas=(0.9,0.98),eps=1e-9))
if __name__=='__main__':
    path='../.data/iwslt/de-en/IWSLT16.TED.dev2010.de-en'
    epoch=10

    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    #prepare dataset
    ds=MyDataSet(path)
    data_iter=MyDataLoader(ds,batch_size=100,shuffle=True,device=device)
    #prepare model
    model=makeModel(ds.srcV(),ds.tgtV())
    model=model.to(device)
    
    optimizer=get_std_opt(model)
    generator=LabelSmoothingLoss(ds.tgtV(),paddingidx=ds.padding_idx,smooth=0.1)
    loss_func=ComputeLoss(generator,optimizer)
    for i in range(epoch):
        run_train_epoch(data_iter,model,loss_func,i+1,display=2)
    #run_eval(data_iter,model)
