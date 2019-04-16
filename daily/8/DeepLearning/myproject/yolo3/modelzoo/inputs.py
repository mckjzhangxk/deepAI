import caffe
import os
import numpy as np
from tqdm import tqdm

def dataIterator(imageprefix,valfile,batchsize,trainsformer=None):

    with open(valfile) as fs:
        while True:
            lines=[fs.readline().strip('\n') for i in range(batchsize)]
            Xs,Ys=[],[]
            end=False
            for line in lines:
                if line=='':
                    end=True
                    break
                sps=line.split(' ')
                imagepath=os.path.join(imageprefix,sps[0])
                if not os.path.exists(imagepath):
                    continue
                X=caffe.io.load_image(imagepath)

                if trainsformer:
                    X=trainsformer.preprocess('input',X)
                Y=int(sps[1])

                Xs.append(X)
                Ys.append(Y)
            Xs=np.array(Xs)
            Ys=np.array(Ys)
            yield Xs, Ys
            if end: break
def getImageNetMean(path):
    u=np.load(path)
    u=np.reshape(u,(3,-1))
    u=np.mean(u,axis=1)
    return u
def cafeDataTransfer(batchsize,mean,shape=(227,227,3)):
    transfomer=caffe.io.Transformer({'input':(batchsize,*shape[::-1])})
    transfomer.set_transpose('input',(2,0,1))
    transfomer.set_mean('input',mean)
    transfomer.set_channel_swap('input',(2,1,0)) #RGB->BGR
    transfomer.set_raw_scale('input',255.0)
    return transfomer

class DataIterator():
    def __init__(self,imageprefix,valfile,batchsize,meanfile,shape):
        self.prefix=imageprefix
        self.valfile=valfile
        self.batchsize=batchsize
        self.transformer=cafeDataTransfer(batchsize,getImageNetMean(meanfile),shape=shape)
        self.examples=50000
    def __iter__(self):
        return dataIterator(self.prefix,self.valfile,self.batchsize,self.transformer)
    def __len__(self):return self.examples//self.batchsize

print(getImageNetMean('/home/zxk/AI/ILSVRC2012/ilsvrc_2012_mean.npy'))
# [103.939, 116.779, 123.68]
# iterator=dataIterator('/home/zxk/AI/ILSVRC2012/ILSVRC2012_img_val','/home/zxk/AI/ILSVRC2012/val.txt',128,transfer)
# db=DataIterator('/home/zxk/AI/ILSVRC2012/ILSVRC2012_img_val',
#                 '/home/zxk/AI/ILSVRC2012/val.txt',128,
#                 '/home/zxk/AI/ILSVRC2012/ilsvrc_2012_mean.npy')

# cnt=0
#
# for x,y in tqdm(db):pass
    # cnt+=len(x)
    # print(cnt)
# for i in tqdm(range(10)):
#     print(i)
