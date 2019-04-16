import caffe
from modelzoo.inputs import DataIterator,cafeDataTransfer,getImageNetMean
import os
from modelzoo.BaseNet import BaseModel
class caffeNet(BaseModel):
    def __init__(self,modelpath):
        super(caffeNet,self).__init__(modelpath,(227,227,3))
    def get_weighs_name(self):
        return 'bvlc_reference_caffenet.caffemodel'


# db=DataIterator('/home/zxk/AI/ILSVRC2012/ILSVRC2012_img_val',
#                 '/home/zxk/AI/ILSVRC2012/val.txt',128,
#                 '/home/zxk/AI/ILSVRC2012/ilsvrc_2012_mean.npy')
# model=caffeNet('/home/zxk/AI/caffe/models/bvlc_reference_caffenet')

# u=getImageNetMean('/home/zxk/AI/ILSVRC2012/ilsvrc_2012_mean.npy')
# transformer=cafeDataTransfer(2,u)
# X=transformer.preprocess('input',caffe.io.load_image('/home/zxk/AI/caffe/examples/images/cat.jpg'))
# print(X.shape)
# caffe.set_mode_gpu()
# for X,y in db:
#     prob=model.forward(X)
#     print(type(prob))
#     a=np.argmax(prob,axis=-1)
#     print(a)