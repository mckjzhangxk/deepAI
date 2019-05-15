import shutil
from tqdm import tqdm
import numpy.random as npr
import numpy as np
import random
import glob
import cv2
import PIL.Image as Image
import os
from threading import Thread,currentThread
from multiprocessing import cpu_count


cnt=0

class MyWorker(Thread):
    def __init__(self,imagelist,tgt,offset):
        super().__init__()
        self.imagelist=imagelist
        self.tgt=tgt
        self.offset=offset
    def run(self):
        name=currentThread().getName()
        print(name,':data size:',len(self.imagelist))
        createDetectDataSet(self.imagelist,self.tgt,offset=self.offset)
def createDetectDataSet(imglist,tgt,num=None,offset=0):
    #imglist=glob.glob(src)
#     random.shuffle(imglist)
    if num:imglist=imglist[:num]
    def decodeBox(imgname):
        I=cv2.imread(imgname)
        H,W,_=I.shape
        W,H=Image.open(imgname).size
        
        basename=os.path.basename(imgname)
        sps=basename.split('-')

        bbox=sps[2]
        bbox=bbox.split('_')
        pp1=bbox[0].split('&')
        pp2=bbox[1].split('&')
        x1,y1,x2,y2=int(pp1[0]),int(pp1[1]),int(pp2[0]),int(pp2[1])
        cx,cy,w,h=(x1+x2)/2,(y1+y2)/2,(x2-x1+1),(y2-y1+1)
        return cx/W,cy/H,w/W,h/H
    for i,imgpath in enumerate(imglist):
        ##open file
        
        cx,cy,w,h=decodeBox(imgpath)
        content=' '.join(map(str,[0,cx,cy,w,h]))
        
        tgt_img=os.path.join(tgt,'%d.jpg'%(offset+i))
        tgt_label=os.path.join(tgt,'%d.txt'%(offset+i))
        shutil.copy(imgpath,tgt_img)
        with open(tgt_label,'w') as fs:
            fs.write(content)
            

src_path='/home/zxk/AI/data/CCPD2019/ccpd_base/*.jpg'
imglist=glob.glob(src_path)
tgt_path='my'
cpus=cpu_count()
BATCH=int(np.ceil(len(imglist)/cpus))
threads=[]
for i in range(cpus):
    t=MyWorker(imglist[i*BATCH:i*BATCH+BATCH],tgt_path,offset=i*BATCH)    
    threads.append(t)
    t.start()
for t in threads:
    t.join()
