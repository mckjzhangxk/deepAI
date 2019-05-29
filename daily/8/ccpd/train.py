import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import CNet,compute_loss
from dataset import CCPD_Dataset
import os
import argparse

class CustomOptimizer():
    def __init__(self,factor,warmup,optimizer):
        self.factor=factor 
        self.warmup=warmup
        self._step=0
        self.optimizer=optimizer
        
    def rate(self):
        return min((self._step*((self.warmup)**(-1.5)))     ,(self._step**(-0.5))) *self.factor
    def step(self):
        self._step+=1
        for p in self.optimizer.param_groups:
            p['lr']=self.rate()
        self.optimizer.step()
    def zero_grad(self):
        self.optimizer.zero_grad()
    def state_dict(self):
        return self.optimizer.state_dict()
    def load_state_dict(self,state):
        self.optimizer.load_state_dict(state)
def get_std_opt(model,factor=2,warmup=4000):
    return CustomOptimizer(factor=factor,warmup=warmup,optimizer=optim.Adam(model.parameters(),lr=0.0,betas=(0.9,0.98),eps=1e-9))
def restore(model,optimizer=None,path=None,tocpu=False):
    import glob


    paths=glob.glob(os.path.join(path,'*.pt'))
    if len(paths)==0:return (0,0)
    paths=sorted(paths,key=lambda x:int(x[x.rfind('E')+1:x.rfind('.')]))
    path=paths[-1]
    checkpoint=torch.load(path) if torch.cuda.is_available() else torch.load(path,map_location='cpu')

    model.load_state_dict(checkpoint['model'])
    step=checkpoint['step']
    epoch=checkpoint['epoch']
    lastloss=checkpoint['loss']

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
        optimizer._step = step

    print('recover model from path:{},epoch {},step {},last loss {}'.format(path,epoch,step,lastloss) )
    return (epoch,lastloss)
def saveModel(model,optimizer,epoch,loss,modelpath):
    r={}
    r['model']=model.state_dict()
    r['optimizer']=optimizer.state_dict()
    r['step']=optimizer._step
    r['epoch']=epoch
    r['loss']=loss

    savepath = os.path.join(modelpath, 'E%d.pt' % epoch)
    torch.save(r,savepath)
    print('save in path:', savepath)


def train_epoch(model, lossfun, ds, optimizer, epoch, display=10, device='cpu'):
    model.train()
    for i, (x, y) in enumerate(ds):
        x = x.to(device)
        y = y.to(device)

        yhat = model(x)
        _, avgloss = lossfun(yhat, y)
        if i % display == 0:
            print('Epoch %d,step:%d,loss:%.4f' % (epoch,i, avgloss))

        with torch.no_grad():
            optimizer.zero_grad()
            avgloss.backward()
            optimizer.step()
    return avgloss
def eval_epoch(model,ds,device='cpu'):
    model.eval()

    totalnum=0
    digital_right=0
    pl_right=0

    for i, (x, y) in enumerate(ds):
        x=x.to(device)
        y=y.to(device)


        yhat=model(x)
        yhat=torch.split(yhat,[34,25,35,35,35,35,35],dim=1)

        ypred=torch.argmax(yhat[0],dim=1,keepdim=True)
        for c in yhat[1:]:
            ypred=torch.cat((ypred,torch.argmax(c,dim=1,keepdim=True)),dim=-1)

        mask=(y==ypred)
        mask1=(mask.sum(dim=1)==7).sum()
        digital_right+=mask.sum().item()
        pl_right+=mask1.sum().item()
        totalnum+=y.size(0)

    acc1=digital_right/(totalnum*7)
    acc2=pl_right/totalnum
    assert acc1>=acc2
    return acc1,acc2


def myParseArgument():
    parser=argparse.ArgumentParser()

    parser.add_argument('--modelpath',type=str,help='use to save and recover model')
    parser.add_argument('--dataset',type=str,help='dataset')
    parser.add_argument('--epoch',type=int,default=15)
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--warmup',type=int,default=4000)
    parser.add_argument('--lr', type=float, default=1e-2)

    return parser.parse_args()

if __name__=='__main__':
    #python3 train.py --dataset=/home/zxk/AI/data/CCPD2019/sample/*.jpg  --modelpath=models --epoch=15 --batch_size=32 --warmup=100 --lr=1e-3
    args=myParseArgument()

    dbpath=args.dataset
    modelpath=args.modelpath
    epoch=args.epoch
    batch_size=args.batch_size
    warmup=args.warmup
    lr=args.lr
    seed=100


    print(args)

    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('using device:',device)

    #prepare dataset
    trainset = CCPD_Dataset(dbpath, True,seed=seed)
    testset = CCPD_Dataset(dbpath, False,seed=seed)
    print('trainset:', len(trainset))
    print('teset:', len(testset))

    trainloader = DataLoader(trainset, batch_size, shuffle=True, num_workers=4)
    testloader = DataLoader(trainset, batch_size, shuffle=False, num_workers=4)


    #prepare model
    
    model=CNet()
    model=model.to(device)
    print('#params:',model.pcount())
    optimizer=get_std_opt(model,factor=lr,warmup=warmup)

    start,_=restore(model,optimizer,modelpath)
    for i in range(start+1,epoch):
        loss=train_epoch(model=model,lossfun=compute_loss,optimizer=optimizer,epoch=i,ds=trainloader,device=device)
        saveModel(model,optimizer,i,loss,modelpath)
        print('Running Eval:')
        acc1,acc2=eval_epoch(model,testloader,device)
        print('accuracy digital:%.2f,accuracy:%.2f'%(acc1,acc2))