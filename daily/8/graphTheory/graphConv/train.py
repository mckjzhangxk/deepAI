import argparse

def parseArgument():
    parser=argparse.ArgumentParser()

    # 数据方面参数
    parser.add_argument('--dataPath',type=str,default='data/cora')
    # 模型方面参数 
    parser.add_argument('--nHidden',type=int,default=128)
    parser.add_argument('--dropout',type=float,default=0.2)

    parser.add_argument('--loss',type='str',default='softmax')
    #训练方面参数
    parser.add_argument('--gpu',action='store_true')
    parser.add_argument('--epoch',type=int,default=20)
    parser.add_argument('--lr',type=float,default=0.01)
    parser.add_argument('--weight_decay',type=float,default=1e-4)

    # 模型存储方面
    parser.add_argument('--savepath',type=str,default='model/graph.pth')
    return parser.parse_args()
def trainGraphConvolutionNetwork(hparam):
    from model import GraphConvNetwork
    from data.dataset import CoraDataSet

    device='cuda' if hparam.gpu else 'cpu'

    db=CoraDataSet(basepath=hparam.dataPath)
    X,y,A,idx_train,idx_val,idx_test=db.getTorchTensor(device)
    
    
    net=GraphConvNetwork(inchannel=db.featureDim,
                            nhidden=hparam.nHidden,
                            outchannel=db.numClass,
                            dropout=hparam.dropout).to(device).train()
    from losses import LossFamily
    lossfunc=LossFamily[hparam.loss]

    import torch.optim as optim
    optimizer=optim.Adam(net.parameters(),lr=hparam.lr,weight_decay=hparam.weight_decay)
    
    from metric import accuracy
    for epoch in range(hparam.epoch):
        optimizer.zero_grad()
        #forward path
        yhat=net(X,A)
        _loss=lossfunc(yhat[idx_train],y[idx_train])
        # backward
        _loss.backward()
        optimizer.step()

        #统计截断
        trainLoss=_loss.item()
        valLoss=lossfunc(yhat[idx_val],y[idx_val]).item()
        
        trainAcc=accuracy(yhat[idx_train],y[idx_train])
        valAcc=accuracy(yhat[idx_val],y[idx_val])

        print('epoch %d,train accuracy %.2f,val accuracy %.2f'%(trainAcc,valAcc))
        torch.save(net.state_dict(),hparam.savepath)
        
if __name__ == "__main__":
    args=parseArgument()
    trainGraphConvolutionNetwork(args)
