import mxnet as mx


class ArcFace():
    def __init__(self,modelpath=None,device=None):
        symbol,arg_params,aux_params=mx.model.load_checkpoint(modelpath,0)
        sym=symbol.get_internals()['fc1_output']
        ctx=mx.cpu() if device=='cpu' else mx.gpu()
        self.model = mx.mod.Module(symbol=sym, context=ctx,label_names=None)

        self.model.bind(data_shapes=[('data', (1, 3, 112, 112))])
        self.model.set_params(arg_params, aux_params)
    def extractFeature(self,input_blob):
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False)
        embedding = self.model.get_outputs()[0].asnumpy()
        
if __name__ == '__main__':
    mymodel=ArcFace('models/model','cpu')