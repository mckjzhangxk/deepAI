import torch
import torch.optim as optim
from model import makeModel,LabelSmoothingLoss,ComputeLoss,run_train_epoch,run_eval 
from dataset import MyDataSet,MyDataLoader
import os
import argparse

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
    def state_dict(self):
        return self.optimizer.state_dict()
    def load_state_dict(self,state):
        self.optimizer.load_state_dict(state)
def get_std_opt(model):
    return CustomOptimizer(model.src_emb[0].d,2,100,optim.Adam(model.parameters(),lr=0.0,betas=(0.9,0.98),eps=1e-9))
def restore(model,optimizer,path):
    import glob


    paths=glob.glob(os.path.join(path,'*.pt'))
    if len(paths)==0:return (0,0)
    paths=sorted(paths,key=lambda x:int(x[x.rfind('E')+1:x.rfind('.')]))
    path=paths[-1]
    checkpoint=torch.load(path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch=checkpoint['epoch']
    optimizer._step=checkpoint['step']
    lastloss=checkpoint['loss']

    print('recover model from path:{},epoch {},step {},last loss {}'.format(path,epoch,optimizer._step,lastloss) )

    return (epoch,lastloss)


def myParseArgument():
    parser=argparse.ArgumentParser()

    parser.add_argument('--modelpath',type=str,help='use to save and recover model')
    parser.add_argument('--trainset',type=str,help='train dataset')
    parser.add_argument('--testset',type=str,help='test dataset')
    parser.add_argument('--epoch',type=int)
    parser.add_argument('--batch_size',type=int)

    return parser.parse_args()

if __name__=='__main__':
#    trainpath='../.data/iwslt/de-en/test'
#    testpath='../.data/iwslt/de-en/test'
#
#    modelpath='./model'
#    epoch=50
#
    #python3 train.py --trainset=../.data/iwslt/de-en/test --testset=../.data/iwslt/de-en/test --modelpath=./model --epoch=50
    args=myParseArgument()
    trainpath=args.trainset
    testpath=args.testset
    modelpath=args.modelpath
    epoch=args.epoch
    batch_size=args.batch_size


    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('using device:',device)

    #prepare dataset
    train_ds=MyDataSet(trainpath)
    train_data_iter=MyDataLoader(train_ds,batch_size=batch_size,shuffle=True,device=device)
    
    test_ds=MyDataSet(testpath)
    test_data_iter=MyDataLoader(test_ds,batch_size=batch_size,shuffle=False,device=device)
    #prepare model
    print('word: src:%d,tgt:%d'%(train_ds.srcV(),train_ds.tgtV()))
    model=makeModel(train_ds.srcV(),train_ds.tgtV())
    model=model.to(device)
    
    optimizer=get_std_opt(model)
    generator=LabelSmoothingLoss(train_ds.tgtV(),paddingidx=train_ds.padding_idx,smooth=0.1)
    loss_func=ComputeLoss(generator,optimizer)
    
    start,_=restore(model,optimizer,modelpath)
    for i in range(start,epoch):
        state=run_train_epoch(train_data_iter,model,loss_func,i,display=2)
        state['optimizer']=optimizer.state_dict()

        savepath=os.path.join(modelpath,'E%d.pt'%i)
        torch.save(state,os.path.join(modelpath,'E%d.pt'%(i)))
        print('save in path:',savepath)
        print('Running Eval:')
        run_eval(test_data_iter,model)
    #run_eval(data_iter,model)
