# from data.data_pipe import de_preprocess, get_train_loader, get_val_data
from arcface_pytorch.model import Backbone, MobileFaceNet,l2_norm
from utils.imageUtils import uniformNormal
import torch

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from torchvision import transforms as trans
import os

class ArcFace(object):
    def __init__(self,conf=None):
        self.device=conf['device']
        self.imagesize=conf['arcface_image_size']
        if conf['arcface_arch']=='mobilefacenet':
            self.model = MobileFaceNet(conf.embedding_size).to(conf.device)
            print('MobileFaceNet model generated')
        else:
            self.model = Backbone(conf['arcface_net_depth'], conf['arcface_drop_ratio'],
                                  conf['arcface_net_mode']).to(conf['device']).eval()
            print('{}_{} model generated'.format(conf['arcface_net_mode'], conf['arcface_net_depth']))
        self.load_state(conf)


    def load_state(self,conf):
        modelpath=conf['arcface_modelpath']
        self.model.load_state_dict(torch.load(modelpath,map_location=self.device))


    def infer(self,faces, tta=False):
        '''
        faces : list of PIL Image
        target_embs : [n, 512] computed embeddings of faces in facebank
        names : recorded names of faces in facebank
        tta : test time augmentation (hfilp, that's all)
        '''
        inputTensor=[]
        inputTensor_tta=[]
        for img in faces:
            img=img.resize((self.imagesize,self.imagesize))
            if tta:
                inputTensor_tta.append(uniformNormal(img,True))
            inputTensor.append(uniformNormal(img))
        inputTensor=torch.stack(inputTensor,dim=0)

        emb = self.model(inputTensor.to(self.device))
        if tta:
            inputTensor_tta = torch.stack(inputTensor_tta, dim=0)
            emb_mirror=self.model(inputTensor_tta.to(self.device))
            emb=l2_norm(emb + emb_mirror)
        else:
            emb=l2_norm(emb)
        return emb
if __name__ == '__main__':
    from config import loadConfig
    config=loadConfig()
    facereg=ArcFace(config)
    from PIL import Image
    import PIL
    from glob import glob
    images=glob('/home/zhangxk/projects/deepAI/daily/8/DeepLearning/myproject/videoAnalysis/data/infinitywar/*.png')

    Is=[]
    for filename in images:
        I=Image.open(filename)
        Is.append(I)

    embs=facereg.infer(Is,False)
    print(embs.shape)