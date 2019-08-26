import torch
import torch.nn as nn

class SoftMaxLoss(nn.Module):
    def __init__(self):
        super().__init__()
        #这个input可以是(batch,C,d1.....dk),期望的输入是logit,是LogSoftmax+NLLLoss的组合
        self.loss=nn.CrossEntropyLoss(reduction='mean')
    def forward(self,logit,target):
        out=self.loss(logit,target)
        return out

LossFamily={
    'softmax':SoftMaxLoss()
}    
if __name__ == "__main__":

    yhat=torch.randn(22,10)
    y=torch.randint(0,10,size=(22,))
    
    ls=SoftMaxLoss()
    xx=ls(yhat,y)
    print(xx)

    logprob=nn.LogSoftmax(1)(yhat)
    xx=nn.NLLLoss(reduction='mean')(logprob,y)
    print(xx.item())