import cv2
import json
import random
from utils import ioa,videoWriter
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from utils import prewhiten
import numpy as np
import glob
import os
import torch
from PIL import ImageFont,ImageDraw,Image
import tqdm

def findSomeBody(model,detectpath,imagesize=160):
    names=[]
    features=[]
    for f in glob.glob(detectpath):
        I = cv2.imread(f)
        I=I[:,:,::-1]
        I = cv2.resize(I, (imagesize, imagesize))
        I = np.transpose(I, (2, 0, 1))
        # I=np.expand_dims(I,0)
        I = prewhiten(I)
        names.append(os.path.basename(f)[0:-4])
        features.append(I)
    I=torch.stack(features,0)
    faceids = model(I).cpu().data.numpy()
    return names,faceids


def dist(a,b):
    r=np.abs((a*b).sum(axis=1))
    a=np.linalg.norm(a,axis=1)
    b =np.linalg.norm(b)
    return r/a/b

def colorSchme(s):
    random.seed(hash(s) % 100000)
    return [random.randint(0, 255) for _ in range(3)]

fontpath = "simsun.ttc" # <== 这里是宋体路径
font = ImageFont.truetype(fontpath, 28)

def putText(img,orgin,color,text=''):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text(orgin, text, font=font,fill=color)
    img = np.array(img_pil)
    return img

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        img1=putText(img,(c1[0], c1[1] - 28),(225, 255, 255),label)
        img[:]=img1
        # cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
def plotImage(obj,img):

    label = obj['type']
    h=label
    boxes = obj['box']

    if 'face_box' in obj:
        h=label+str(obj['face_box'])
    color=colorSchme(h)
    plot_one_box(boxes, img, color, label)
    if 'face_box' in obj:
        face_box=obj['face_box']
        d=dist(feature,np.array(obj['face_id']))
        score=np.max(d)
        score_index=np.argmax(d)

        plot_one_box(face_box, img, color,'%s,%.2f'%(name[score_index],float(score)))
def resize(img,screen=(1440,900)):
    H,W,_=img.shape
    sw,sh=screen
    if (sw*H/W>sh):
        w,h=sh*W/H,sh
    else:
        w,h=sw,sw*H/W
    return cv2.resize(img,(int(w),int(h)))
def debug(videopath,jsonpath):
    cap = cv2.VideoCapture(videopath)
    
    with open(jsonpath,'r') as fs:
        d=json.load(fs)

    ts=d['track']['ts']
    objs=d['track']['objs']

    step=0
    writer=videoWriter('output.avi',videoformat='XVID',scale=(int(cap.get(3)),int(cap.get(4))),fps=3)

    for step in tqdm.tqdm(range(len(ts))):
    # for t,objts in zip(ts,objs):
        t=ts[step]
        objts=objs[step]
        cap.set(1,t)
        ret, img = cap.read()
        if not ret:break

        for obj in objts:
            plotImage(obj,img)
        # img=resize(img)
        # cv2.imshow("input", img)
        # key = cv2.waitKey()
        # if key == ord('q'):
        #     break
        # elif key==83:
        #     step=min(step+1,len(ts)-1)
        # elif key==81:
        #     step=max(step-1,0)
        writer.write(img)
    writer.release()
    cap.release()


model=InceptionResnetV1(pretrained='casia-webface').eval()
name,feature=findSomeBody(model,'data/infinitywar/*.png')
# name,feature=findSomeBody(model,'data/spiderman/*.png')
if __name__ == '__main__':
    # debug('data/spman.avi','tmp/input/final/spman.json')
    debug('data/war.avi', 'tmp/input/final/war.json')