import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConv(nn.Module):
    def __init__(self,inchannel,outchannel):
        super().__init__()
        self.weight=nn.Parameter(torch.zeros((inchannel,outchannel)),requires_grad=True)
        nn.init.xavier_normal_(self.weight)

        self.bias=nn.Parameter(torch.zeros((outchannel)),requires_grad=True)
        # self.register_parameter('xx',self.weight)
        # self.register_parameter('bias',self.bias)
    def forward(self,X,A):
        '''
            A是稀疏矩阵 (V,V)
            X是节点的特征（V，inchannel)
            输出(V,outchannel)
        '''
        f=torch.spmm(A,X)
        out=torch.mm(f,self.weight)+self.bias
        return out
class GraphConvNetwork(nn.Module):
    def __init__(self,inchannel,nhidden,outchannel,dropout=0.8):
        super().__init__()
        self.conv1=GraphConv(inchannel,nhidden)
        self.conv2=GraphConv(nhidden,outchannel)
        self.drop=nn.Dropout(p=dropout)
    def forward(self,X,A):
        import torch.nn.functional as F
        X=self.conv1(X,A)
        X=F.relu(X)
        
        x=self.drop(X)

        X=self.conv2(X,A)

        return X
if __name__ == "__main__":
    import numpy as np 
    from scipy.sparse import coo_matrix
    
    device='cpu'
    V,F=10,3
    nhidden=5


    Adense=np.random.randint(0,1,size=(V,V)).astype(np.int32)
    Asparse=coo_matrix(Adense)

    row=Asparse.row
    col=Asparse.col
    val=Asparse.data

    ii=torch.Tensor(np.vstack((row,col))).long().to(device)
    val=torch.from_numpy(val).float().to(device)
    A=torch.sparse.FloatTensor(indices=ii,values=val,device='cpu',size=(V,V))
    X=torch.rand(V,F).to(device)

    # conv=GraphConv(F,nhidden).to(device)
    # print(list(conv.parameters()))
    # out=conv(X,A)
    # print(out.dtype,out.shape,out.device)

    net=GraphConvNetwork(F,nhidden,7).to(device).train()
    O=net(X,A)
    print(O.shape,O.dtype,O.device)
    # print(net.conv1.weight.dtype,net.conv1.weight.shape,net.conv1.weight.device)
    # print(net.conv2.weight.dtype,net.conv2.weight.shape,net.conv2.weight.device)

    # net=net.to(device)
    # print(net.conv1.weight.dtype,net.conv1.weight.shape,net.conv1.weight.device)
    # print(net.conv2.weight.dtype,net.conv2.weight.shape,net.conv2.weight.device)
    # print(net)
    # for f in net.parameters():
    #     print(f.shape,f.dtype,f.device)
    #     # print(f)