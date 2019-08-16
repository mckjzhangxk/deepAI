import mxnet.recordio as io
import numpy as np
import cv2
import os
from mtcnn_pytorch import MTCNN
from utils.alignUtils import AlignFace
from arcface import ArcFace


def dumpFeature(prefix,model,outdir,batch=64,flush=5):
    from pathlib import Path
    p=Path(outdir)
    if p.exists():
        import shutil
        shutil.rmtree(outdir)
        p.mkdir()

    reader=io.MXIndexedRecordIO(prefix+'.idx',prefix+'.rec','r')

    #第0行是全部种类的信息，获得全部种类的索引
    s=reader.read_idx(0)
    header,_=io.unpack(s)
    labels=range(int(header.label[0]),int(header.label[1]))
    ###############获得种类下实例的索引，imgs保存的是某一个种类下的实例索引####
    imgs=[]
    for l in labels:
        s=reader.read_idx(int(l))
        header,_=io.unpack(s)
        a,b=int(header.label[0]),int(header.label[1])
        imgs.append(range(a,b))
    queue=[]
    from bcolz import carray

    features_label=carray([],rootdir=os.path.join(outdir,'label'))
    features=carray(np.empty((0,512)),rootdir=os.path.join(outdir,'feature'))



    ##########extract feature of every image##############
    import tqdm
    steps=0
    for imgidxs in tqdm.tqdm(imgs):
        for id in tqdm.tqdm(imgidxs):
            s=reader.read_idx(id)
            h,img=io.unpack_img(s)

            queue.append(model.processInput(img[:,:,::-1]))
            features_label.append(h.label)

            if len(queue)==batch:
                ########################
                f=model.handle(queue)
                features.append(f)
                ########################
                del queue
                queue=[]
            if steps>flush and steps%flush==0:
                features_label.flush()
                features.flush()
            steps+=1
    features_label.flush()
    features.flush()

class Model():
    def __init__(self,imgsize = 112,device = 'cuda'):
        self.det = MTCNN(device)
        self.alignutils = AlignFace((imgsize, imgsize))
        # self.model=InceptionResnetV1(pretrained='casia-webface').to(device).eval()
        self.model = ArcFace('arcface/models/model', device, imgsize)
        self.imgsize=imgsize
        self.device=device
    def processInput(self,img):
        '''
        
        :param img:numpy RGB格式 
        :return: 
        '''
        import PIL

        faceboxes, landmarks = self.det.detect_faces(PIL.Image.fromarray(img))

        if len(faceboxes)>0:
            frame=self.alignutils.align(PIL.Image.fromarray(img),
                                  faceboxes[0][:4],landmarks[0],
                                  40,
                                  0)
        else:frame=PIL.Image.fromarray(img)
        I = cv2.resize(np.array(frame), (self.imgsize, self.imgsize))
        return self.model._preprocessing(I)
    def handle(self,queue):
        r=self.model.extractFeature(queue,self.device)
        return np.array(r)

if __name__ == '__main__':
    prefix = '/home/zhangxk/AIProject/数据集与模型/arcface_dataset/faces_umd/train'
    root_dir='dump'
    device = 'cpu'
    imgsize = 112
    model=Model(imgsize=imgsize,device=device)
    dumpFeature(prefix, model, root_dir, batch=4, flush=5)

