import cv2
import numpy as np
from align.detect_face import detect_face,create_mtcnn
import tensorflow as tf
import matplotlib.pyplot as pyplot
import os
from openfaceModel import getModel

sess=tf.Session()
pnet_fun, rnet_fun, onet_fun = create_mtcnn(sess, None)
model=getModel()

def face_encodings(filename):
    I=_detectSingleFace(filename)
    if I is None:return []

    if I.shape!=4:
        I.shape=(1,96,96,3)
    code=model.predict(I)
    return [code[0]]

def _detectSingleFace(path, minsize=50, threshold = [0.6, 0.7, 0.7], factor=0.709,pad=5):
    I = cv2.imread(path)
    I = I[:, :, ::-1]

    box, point = detect_face(I, minsize, pnet_fun, rnet_fun, onet_fun, threshold, factor)
    if len(box) == 0:
        return None

    x1, y1, x2, y2, acc = box[0]
    x1, y1, x2, y2 = max(x1-pad, 0), max(y1-pad, 0), max(x2+pad, 0), max(y2+pad, 0)

    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    h,w=y2-y1,x2-x1
    print(h,w)

    newI = I[y1:y2, x1:x2]

    newI = cv2.resize(newI, (96, 96))
    # zeroI=np.zeros((96,96,3),dtype=np.uint8)+255
    # c=zeroI.shape[1]
    # c1=newI.shape[1]
    # s=(c-c1)//2
    # zeroI[:,s:s+c1,:]=newI
    #
    a,b=os.path.split(path)
    pyplot.imsave('test/'+b,newI)
    return newI/255.0

def detectFace(path='facedb',sess=sess,
               minsize=50,
               threshold = [ 0.6, 0.7, 0.7 ],
               factor=0.709):
    '''

    :param path:
    :param sess:
    :return:a list of Image,None stand for can not detectFace
    '''
    ret=[]
    for f in os.listdir(path):
        I=cv2.imread(os.path.join(path,f))
        I=I[:,:,::-1]

        box,point=detect_face(I,minsize,pnet_fun,rnet_fun,onet_fun,threshold,factor)
        if len(box)==0:
            ret.append(None)
            continue

        x1, y1, x2, y2,acc=box[0]
        x1,y1,x2,y2=max(x1,0),max(y1,0),max(x2,0),max(y2,0)

        x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
        newI=I[y1:y2, x1:x2]
        newI=cv2.resize(newI,(96,96))
        ret.append((f,newI))
    return ret