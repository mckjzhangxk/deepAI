#################################################
#1. I will load face image,detect it's feature
#2. compute feature's index
#3. then generate a record for each picture

#id filename feature id0 id1 .... id 255

#################################################
from arcface import ArcFace
import cv2
import numpy as np
import glob
from search.pq import quanizer
import tqdm

mymodel=ArcFace('../arcface/models/model','cpu')
basepath='/home/zhangxk/AIProject/数据集与模型/arcface_dataset/faces_umd/db'
def getFeature(imagepath):
    im=cv2.imread(imagepath)
    im=cv2.resize(im,(112,112))
    im=im[:,:,::-1]
    im=np.expand_dims(im,0)
    im = np.transpose(im, (0, 3, 1, 2))

    emb=mymodel.forward(im)
    return emb[0]

def readFiles():
    import os
    filenames=glob.glob(os.path.join(basepath,'*/*.jpg'))
    return filenames
class LSH():
    def __init__(self,cookbook='codebook.npy'):
        self.cookbook=np.load(cookbook)
    def localHash(self,feature):
        return quanizer(feature,self.cookbook)[0].tolist()
def record(filename,emb,index):
    import uuid
    id=str(uuid.uuid1()).replace('-','')

    emb=str(emb.tolist())
    index=','.join(map(str,index))
    return filename+' '+index

if __name__ == '__main__':
    filelist=readFiles()
    lsh=LSH()

    filelist=filelist[:3]
    with open('record.sql','w') as fs:
        for filename in tqdm.tqdm(filelist):
            emb=getFeature(filename)
            index=lsh.localHash(emb)
            s=record(filename,emb,index)
            fs.write(s+'\n')
