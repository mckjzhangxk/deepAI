import torch
import torch.nn as nn
import torch.nn.functional as F
class BlockLayer(nn.Module):
    def __init__(self,in_ch,out_ch,stride=1):
        super(BlockLayer,self).__init__()
        H=out_ch//4
        model=nn.Sequential()
        model.add_module('conv1',nn.Conv2d(in_ch,H,3,1,1,bias=False))
        model.add_module('bn1',nn.BatchNorm2d(H))
        model.add_module('relu1',nn.ReLU())
        
        model.add_module('conv2',nn.Conv2d(H,H,3,stride,1,bias=False))
        model.add_module('bn2',nn.BatchNorm2d(H))
        model.add_module('relu2',nn.ReLU())
        
        model.add_module('conv3',nn.Conv2d(H,out_ch,3,1,1,bias=False))
        model.add_module('bn3',nn.BatchNorm2d(out_ch))
        model.add_module('relu3',nn.ReLU())
        if in_ch==out_ch and stride==1:
            self.shortcut=nn.Identity()
        else:
            self.shortcut=nn.Conv2d(in_ch,out_ch,3,stride,1)
        self.block=model
    def forward(self,X):
        route1=self.block(X)
        route2=self.shortcut(X)
        return route1+route2
class GroupLayer(nn.Module):
    def __init__(self,in_ch,out_ch,count):
        super(GroupLayer,self).__init__()
        model=nn.Sequential()
        model.add_module('p1',BlockLayer(in_ch,out_ch,2))
        for i in range(2,count+1):
            model.add_module('p%d'%i,BlockLayer(out_ch,out_ch,1))
        self.model=model
    def forward(self,X):
        return self.model(X)
class CNet(nn.Module):
    def __init__(self):
        super(CNet,self).__init__()
        model=nn.Sequential()
        
        model.add_module('c1',nn.Conv2d(1,64,3,2,1,bias=False))
        model.add_module('b1',nn.BatchNorm2d(64))
        model.add_module('r1',nn.ReLU())
        
        model.add_module('stage1',GroupLayer(64,128,2))
        model.add_module('stage2',GroupLayer(128,256,2))
        model.add_module('stage3',GroupLayer(256,512,2))
        
        self.features=model
        self.cls=nn.Linear(512,34+25+35*5)
#         self.cls2=nn.Linear(512,25)
#         self.cls3=nn.Linear(512,35*5)
        
    def forward(self,X):
        Y=self.features(X)
        Y=F.adaptive_avg_pool2d(Y,(1,1))
        Y=Y.view(-1,512)
        
        Yhat=self.cls(Y)
        
        return Yhat
    def pcount(self):
        c=0
        for x in self.parameters():
            c+=x.numel()
        return c
def compute_loss(Yhat,Y):
    Yhats=torch.split(Yhat,[34,25,35,35,35,35,35],1)
    Ys=torch.split(Y,1,1)
    
    loss=[0,0,0,0,0,0,0]
    for i,(y,yhat) in enumerate(zip(Ys,Yhats)):
        loss[i]+=nn.CrossEntropyLoss()(yhat,y[:,0])
    total_loss=0
    for l in loss:total_loss+=l
    batch_size=Yhat.size(0)
    
    return loss,total_loss/batch_size


if __name__ == '__main__':

    # X=torch.randn(16,1,32,48)
    # G1 = CNet()
    # X1 = G1(X)
    # print(G1.pcount())
    # print(X1.shape)
    yhat=torch.randn((32,10))
    y=torch.randint(low=0,high=10,size=(32,))
    print(y.size())
    print(yhat.size())
    nn.CrossEntropyLoss()(yhat, y)
