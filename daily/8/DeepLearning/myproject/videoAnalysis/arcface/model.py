import mxnet as mx
import numpy as np

class ArcFace():
    def __init__(self,modelpath=None,device=None,imgsize=112):
        symbol,arg_params,aux_params=mx.model.load_checkpoint(modelpath,0)
        sym=symbol.get_internals()['fc1_output']
        self.ctx=mx.cpu() if device=='cpu' else mx.gpu()

        self.model = mx.mod.Module(symbol=sym, context=self.ctx,label_names=None)
        self.imgsize=imgsize

        self.model.bind(data_shapes=[('data', (1, 3, imgsize, imgsize))])
        self.model.set_params(arg_params, aux_params)
    def forward(self,x):
        # self.model.bind(data_shapes=[('data', (len(x), 3, self.imgsize, self.imgsize))])
        x=mx.nd.array(x,self.ctx)
        db = mx.io.DataBatch(data=(x,))
        self.model.forward(db, is_train=False)
        embedding = self.model.get_outputs()[0].asnumpy()
        embedding_norm =np.linalg.norm(embedding,axis=1,keepdims=1)
        embedding=embedding/embedding_norm
        return embedding
    def extractFeature(self,queue,device=None):


        if isinstance(queue,list):
            data = np.stack(queue, 0)
        data=np.transpose(data,(0,3,1,2))
        emb=self.forward(data)
        return emb.tolist()

    def _preprocessing(self,frame):
        I=np.array(frame)
        I=I[:,:,::-1]
        return I
if __name__ == '__main__':
    import numpy as np
    mymodel=ArcFace('models/model','cuda')


    for i in range(2):
        data = np.random.randn(4*(i+1), 3, 112, 112)
        x=mymodel.forward(data)

        print(x.shape)
        print(np.linalg.norm(x,axis=1))