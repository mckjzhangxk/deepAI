import mxnet.recordio as io
import numpy as np
import cv2
import os

prefix='/home/zhangxk/AIProject/数据集与模型/arcface_dataset/faces_umd/train'


def dumpFeature(prefix,model,outdir,batch=64,chucksize=1000):
    reader=io.MXIndexedRecordIO(prefix+'.idx',prefix+'.rec','r')
    s=reader.read_idx(0)
    header,_=io.unpack(s)
    labels=range(int(header.label[0]),int(header.label[1]))

    imgs=[]
    for l in labels:
        s=reader.read_idx(int(l))
        header,_=io.unpack(s)
        a,b=int(header.label[0]),int(header.label[1])
        imgs.append(range(a,b))
    import tqdm
    queue=[]
    features=[]
    chuckidx=0
    for imgidxs in imgs:
        for id in imgidxs:
            s=reader.read_idx(id)
            h,img=io.unpack_img(s)
            if len(queue)==batch:
                ########################

                # features.append(model())
                ########################
                del queue
                queue=[]
            if len(features)>=chucksize:
                outfeature=np.array(features[:chucksize])
                np.save(os.path.join(outdir,'db'%chuckidx),outfeature)
                features=features[chucksize:]
                chuckidx+=1