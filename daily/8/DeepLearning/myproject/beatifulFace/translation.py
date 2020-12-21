import dlib
import cv2
import numpy as np
from faceDection import *
from collections import OrderedDict
from alignUtils import tps_wrap
# import thinplate as tps

FACIAL_LANDMARKS_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 35)),
	("jaw", (0, 17))
])

def warp_image_cv(img, c_src, c_dst, dshape=None):
    dshape = dshape or img.shape
    theta = tps.tps_theta_from_points(c_src, c_dst, reduced=True)
    grid = tps.tps_grid(theta, c_dst, dshape)
    mapx, mapy = tps.tps_grid_to_remap(grid, img.shape)
    return cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC)

def scale(s):
    a=np.min(s,axis=0)
    b=np.max(s,axis=0)
    return (s-a)/(b-a)
def restore(pts,dsize):
    return np.int64(pts*np.float32(dsize))
def drawPoint(img,pts,color=(0,255,0)):
    # img=img.copy()
    for i in range(0,len(pts)):
        x,y=tuple(pts[i])
        cv2.circle(img,(x,y),1,color,1)
        # cv2.putText(img,str(i),(x-10,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.35,color,lineType=2)
    return img


def getGoodRegion(bbox,pts):

    xmin,ymin=bbox[0],bbox[1]
    xmax,ymax=xmin+bbox[2],ymin+bbox[3]

    pts_min=np.min(pts,axis=0)
    pts_max=np.max(pts,axis=0)

    xmin=min(xmin,pts_min[0])
    ymin=min(ymin,pts_min[1])

    xmax=max(xmax,pts_max[0])
    ymax=max(ymax,pts_max[1])

    return xmin,xmax,ymin,ymax

def scale_jaw(jar_pts,s=1.0):
    u=np.mean(jar_pts,axis=0)
    u[1]=0

    return (jar_pts-u)*np.float32([s,1])+u

def readDefaultTemplate(face1_template,filename):
    bbox,shape=face1_template['bbox'],face1_template['feature']

    H,W=face1_template['h'],face1_template['w']
    # image one
    pts1=shape/np.float32([W,H])
    pts1_max=np.max(pts1,axis=0)
    pts1_min=np.min(pts1,axis=0)

    s=pts1_max-pts1_min
    b=pts1_min

    # stand image
    std_pts=np.load(filename)['meanShape']
    std_pts=scale(std_pts)#[0-1]
    std_pts=s*std_pts+b

    return scale(std_pts)

def normalizePoint(face1):
    shape=face1['feature']

    H,W=face1['h'],face1['w']
    
    pts1=shape/np.float32([W,H])
    pts1_max=np.max(pts1,axis=0)
    pts1_min=np.min(pts1,axis=0)

    s=pts1_max-pts1_min
    b=pts1_min

    return pts1,s,b

def getWrapPoints(face1,std_pts):
    pts1,s,b=normalizePoint(face1)
    pts2=std_pts=s*std_pts+b #根据face1的人脸位置，标准的位置
    return pts1,pts2

def  thinScale(pts1,std_pts,s=1.0):
    pts2=pts1.copy()    
    ii,std_jaw_pts=getJaw(std_pts)
    
    pts2[ii]=scale_jaw(std_jaw_pts,s)
    return pts2
def thinFace(face1,std_pts,s=1.0):
    pts1,std_pts=getWrapPoints(face1,std_pts)
    pts2=thinScale(pts1,std_pts,s)

    xmin,xmax,ymin,ymax=getGoodRegion(face1['bbox'],face1['feature'])
    img_result=tps_wrap(face1['image'],pts1,pts2,mask=(xmin,xmax,ymin,ymax))
    return img_result

from datetime import datetime
from sunday import readFace,ABC

if __name__ == '__main__':
    imgpath='data/face5.jpeg'

    face1=readFace(imgpath)
    std_pts=readDefaultTemplate(face1,'data/meanFaceShape.npz')
 

    s=1.0
    img_result=thinFace(face1,std_pts,1.0)
    cv2.imshow('demo2',img_result)
    
    while True:
        key=cv2.waitKey(0)

        if key==ord('q'):break

        if key==ord('a'):
            s+=0.01

        if key==ord('z'):
            s+=-0.01
            
        img_result=thinFace(face1,std_pts,s)

        img_mo=cv2.bilateralFilter(img_result,15,35,35)
        img_result=cv2.cvtColor(ABC(face1['image'],img_result,img_mo),cv2.COLOR_RGB2BGR)
        
        cv2.imshow('demo2',img_result)