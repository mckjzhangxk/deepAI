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

    return parser.parse_args()
def trainGraphConvolutionNetwork(hparam):
    from model import GraphConvNetwork
    from data.dataset import CoraDataSet

    device='cuda' if hparam.gpu else 'cpu'

    db=CoraDataSet(basepath=hparam.dataPath)
    X,y,A=db.getTorchTensor(device)
    
    
    model=GraphConvNetwork(inchannel=db.featureDim,
                            nhidden=hparam.nHidden,
                            outchannel=db.numClass,
                            dropout=hparam.dropout).to(device).train()

    for epoch in range(hparam.epoch):
        yhat=model(X,A)
        print(yhat.shape,yhat.device,yhat.dtype)
if __name__ == "__main__":
    args=parseArgument()
    trainGraphConvolutionNetwork(args)
