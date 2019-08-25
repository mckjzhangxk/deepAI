import torch
import torch.nn as nn

class GraphConv(nn.Module):
    def __init__(self,inchannel,outchannel,device='cpu'):
        super().__init__()
        self.weight=torch.zeros((inchannel,outchannel),requires_grad=True).to(device)
        nn.init.xavier_normal_(self.weight)
        self.bias=torch.zeros((outchannel),requires_grad=True).to(device)
    def forward(self,A,X):
        '''
            A是稀疏矩阵 (V,V)
            X是节点的特征（V，inchannel)
            输出(V,outchannel)
        '''
        pass
        
if __name__ == "__main__":
    
    c=GraphConv(20,30)
    print(c)