import cv2
import os
import numpy as np
from tensorpack.dataflow import imgaug

def dataIterator(imageprefix,valfile,batchsize):
    augmentors = [
        imgaug.ResizeShortestEdge(256, interp=cv2.INTER_CUBIC),
        imgaug.CenterCrop((224, 224)),
    ]
    aug = imgaug.AugmentorList(augmentors)
    with open(valfile) as fs:
        while True:
            lines=[]
            for i in range(batchsize):
                l=fs.readline().strip('\n')
                if l=='':break
                lines.append(l)
            if len(lines)==0:return
            Xs,Ys=[],[]
            for line in lines:
                sps=line.split(' ')
                imagepath=os.path.join(imageprefix,sps[0])
                X=cv2.imread(imagepath)
                assert X is not None ,imagepath
                Y=int(sps[1])

                Xs.append(aug.augment(X))
                Ys.append(Y)
            Xs=np.array(Xs)
            Ys=np.array(Ys)
            yield Xs, Ys
class DataIterator():
    def __init__(self,imageprefix,valfile,batchsize):
        self.prefix=imageprefix
        self.valfile=valfile
        self.batchsize=batchsize
        self.examples=50000
    def __iter__(self):
        return dataIterator(self.prefix,self.valfile,self.batchsize)
    def __len__(self):return self.examples//self.batchsize

# from tqdm import tqdm
# # # [103.939, 116.779, 123.68]
# # # iterator=dataIterator('/home/zxk/AI/ILSVRC2012/ILSVRC2012_img_val','/home/zxk/AI/ILSVRC2012/val.txt',128,transfer)
# db=DataIterator('/home/zxk/AI/ILSVRC2012/ILSVRC2012_img_val',
#                 '/home/zxk/AI/ILSVRC2012/val1.txt',4)
# cnt=0
# for x,y in tqdm(db):
#     print(x.shape)
#     cnt+=len(x)
# print(cnt)



