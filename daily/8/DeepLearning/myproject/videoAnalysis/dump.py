import mxnet.recordio as io
import numpy as np
import cv2
import os
from mtcnn_pytorch import MTCNN
from utils.alignUtils import AlignFace
from arcface import ArcFace


prefix='/home/zhangxk/AIProject/数据集与模型/arcface_dataset/faces_umd/train'


def dumpFeature(prefix,model,outdir,batch=64,chucksize=1000):
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
    import tqdm
    queue=[]
    features=[]
    chuckidx=0
    for imgidxs in imgs:
        for id in imgidxs:
            s=reader.read_idx(id)
            h,img=io.unpack_img(s)

            queue.append(model.processInput(img))

            if len(queue)==batch:
                ########################

                # features.append(model())
                ########################
                del queue
                queue=[]
            if len(features)>=chucksize:
                outfeature=np.array(features[:chucksize])
                np.save(os.path.join(outdir,'chunck%d'%chuckidx),outfeature)
                features=features[chucksize:]
                chuckidx+=1
    if len(features)>0:
        outfeature = np.array(features)
        np.save(os.path.join(outdir, 'chunck%d' % chuckidx), outfeature)
class Model():
    def __init__(self,imgsize = 112,device = 'cuda'):
        self.det = MTCNN(device)
        self.alignutils = AlignFace((imgsize, imgsize))
        # self.model=InceptionResnetV1(pretrained='casia-webface').to(device).eval()
        self.model = ArcFace('arcface/models/model', device, imgsize)
        self.imgsize=imgsize

    def processInput(self,img):
        '''
        
        :param img:numpy RGB格式 
        :return: 
        '''
        import PIL

        faceboxes, landmarks = self.det.detect_faces(PIL.Image.fromarray(img))
        frame=self.alignutils.align(PIL.Image.fromarray(img),
                              faceboxes[0][:4],landmarks[0],
                              40,
                              0)
        I = cv2.resize(np.array(frame), (self.imgsize, self.imgsize))
        return self.model._preprocessing(I)
    def handle(self,queue):
        pass

if __name__ == '__main__':
    # device = 'cuda'
    # imgsize = 112
    # det = MTCNN(device)
    # alignutils = AlignFace((imgsize, imgsize))
    # model=InceptionResnetV1(pretrained='casia-webface').to(device).eval()
    # model = ArcFace('arcface/models/model', device, imgsize)
    pass