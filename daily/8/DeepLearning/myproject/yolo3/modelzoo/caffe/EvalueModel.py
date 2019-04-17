import caffe
import numpy as np
from modelzoo.AlexNet import alexNet
from modelzoo.CaffeNet import caffeNet
from modelzoo.GoogleNet import googleNet
from modelzoo.ResNet import resNet50, resNet101, resNet152
from modelzoo.VGG16Net import vgg16
from tqdm import tqdm

from modelzoo.caffe.inputs import DataIterator

'''
=====================AlexNet=====================
top1 Accuracy:55.79%,top5 Accuracy:79.12%
================================================

=====================CaffeNet=====================
top1 Accuracy:55.98%,top5 Accuracy:79.39%
================================================

=====================Vgg16=====================
top1 Accuracy:65.77%,top5 Accuracy:86.65%
================================================

=====================GoogleNet=====================
top1 Accuracy:68.03%,top5 Accuracy:88.47%
================================================

=====================ResNet50=====================
top1 Accuracy:70.60%,top5 Accuracy:89.88%
================================================

=====================ResNet101=====================
top1 Accuracy:72.23%,top5 Accuracy:90.80%
================================================


=====================ResNet152=====================
top1 Accuracy:72.92%,top5 Accuracy:90.64%
================================================
'''

MODEL_PATH={
    'AlexNet':'/home/zxk/AI/caffe/models/bvlc_alexnet',
    'CaffeNet':'/home/zxk/AI/caffe/models/bvlc_reference_caffenet',
    'GoogleNet':'/home/zxk/AI/caffe/models/bvlc_googlenet',
    'Vgg16':'/home/zxk/AI/caffe/models/vgg16',
    'Res50':'/home/zxk/AI/caffe/models/resnet/res50',
    'Res101':'/home/zxk/AI/caffe/models/resnet/res101',
    'Res152':'/home/zxk/AI/caffe/models/resnet/res152'
}
EVAL_INFORMATION={
    'IMAGE_PREFIX':'/home/zxk/AI/ILSVRC2012/ILSVRC2012_img_val',
    'VAL_FILE':'/home/zxk/AI/ILSVRC2012/val.txt',
    'IMAGENET_MEAN_FILE':'/home/zxk/AI/ILSVRC2012/ilsvrc_2012_mean.npy',
    'BATCH_SIZE':4
}
def showResult(netName,top1,top5,num):
    print()
    print('=====================%s====================='%netName)
    print('top1 Accuracy:%.2f%%,top5 Accuracy:%.2f%%' % (100*top1/num, 100*top5/num))
    print('================================================')
def evalue(yhat,y,K=5):
    '''

    :param yhat: (batch,1000)
    :param y:(batch,)
    :return:
    '''

    top1_result=np.argmax(yhat,-1)
    top1_good=np.sum(top1_result==y)

    y = np.expand_dims(y, 1) #(batch,1)
    topk_result=np.argsort(-yhat,-1)[:,:K] #(batch,K)

    topk_good=np.sum(y==topk_result)

    return top1_good,topk_good,len(y)

def getEvalModel(name='caffeNet'):
    fn=None
    shape=None

    if name=='CaffeNet':
        shape=(227,227,3)
        fn=caffeNet
    elif name=='GoogleNet':
        shape=(224,224,3)
        fn=googleNet
    elif name=='AlexNet':
        shape=(227,227,3)
        fn=alexNet
    elif name=='Vgg16':
        shape=(224,224,3)
        fn=vgg16
    elif name=='Res50':
        shape=(224,224,3)
        fn=resNet50
    elif name=='Res101':
        shape=(224,224,3)
        fn=resNet101
    elif name=='Res152':
        shape=(224,224,3)
        fn=resNet152

    else:
        raise ValueError('Known Model')


    db=DataIterator(EVAL_INFORMATION['IMAGE_PREFIX'],
                    EVAL_INFORMATION['VAL_FILE'],
                    EVAL_INFORMATION['BATCH_SIZE'],
                    EVAL_INFORMATION['IMAGENET_MEAN_FILE'],
                    shape)

    model = fn(MODEL_PATH[name])

    return db,model

if __name__ == '__main__':
    model_name= 'Res152'
    caffe.set_mode_gpu()
    db,model =getEvalModel(name=model_name)

    top1,top5=0,0
    for X,y in tqdm(db):
        probs=model.forward(X)
        _top1,_top5,_=evalue(probs,y)
        top1+=_top1
        top5+=_top5
        print(top1/db.examples,top5/db.examples)
    showResult(model_name, top1, top5, db.examples)