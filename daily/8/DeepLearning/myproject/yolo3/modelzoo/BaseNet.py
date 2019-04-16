import os
import caffe

class BaseModel():
    def __init__(self,modelpath,inputShape):
        self.shape=inputShape

        moded_def=os.path.join(modelpath,'deploy.prototxt')
        model_weight=os.path.join(modelpath, self.get_weighs_name())
        self.net=caffe.Net(moded_def,model_weight,caffe.TEST)
    def get_weighs_name(self):
        raise NotImplemented

    def forward(self,X):
        if X.ndim==3:
            self.net.blobs['data'].data[...]=X
        elif X.ndim==4:
            self.net.blobs['data'].reshape(len(X),self.shape[2],self.shape[0],self.shape[1])
            self.net.blobs['data'].data[...] = X

        else:raise ValueError('input must have dim 3 or 4')
        output=self.net.forward()
        probs=output['prob']

        return probs