import cv2
import os
import numpy as np

def dataIterator(imageprefix,valfile,batchsize):

    with open(valfile) as fs:
        while True:
            lines=[]
            for i in range(batchsize):
                l=fs.readline().strip('\n')
                if l=='':break
                lines.append(l)
            Xs,Ys=[],[]
            for line in lines:
                sps=line.split(' ')
                imagepath=os.path.join(imageprefix,sps[0])
                if not os.path.exists(imagepath):
                    continue
                X=cv2.imread(imagepath)
                Y=int(sps[1])

                Xs.append(X)
                Ys.append(Y)
            Xs=np.array(Xs)
            Ys=np.array(Ys)
            yield Xs, Ys

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


class DataIterator():
    def __init__(self,imageprefix,valfile,batchsize):
        self.prefix=imageprefix
        self.valfile=valfile
        self.batchsize=batchsize
        self.examples=50000
    def __iter__(self):
        return dataIterator(self.prefix,self.valfile,self.batchsize)
    def __len__(self):return self.examples//self.batchsize
