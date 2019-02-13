import tensorflow as tf
import numpy as np
import cv2
from detect import Detector_tf,drawDectectBox,drawLandMarks

def bigPicture(a,b,c,d,h=None,w=None):
    if h:
        a=cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
        b=cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)
        c=cv2.resize(c, (w, h), interpolation=cv2.INTER_AREA)
        d=cv2.resize(d, (w, h), interpolation=cv2.INTER_AREA)
    else:
        h,w,_=a.shape
    out=np.zeros((2*h,2*w,3),dtype=np.uint8)
    out[0:h,0:w] =a
    out[0:h,w:2*w]  =b
    out[h:2*h,0:w]  =c
    out[h:2*h,w:2*w]=d
    return out


pnet_path='/home/zhangxk/AIProject/MTCNN_TRAIN/pnet/model/PNet-29'
rnet_path='/home/zhangxk/AIProject/MTCNN_TRAIN/rnet/model/RNet-23'
onet_path='/home/zhangxk/AIProject/MTCNN_TRAIN/onet/model/ONet-29'
# rnet_path=None
# onet_path=None

sess=tf.Session()
df=Detector_tf(sess,
                minsize=50,
                scaleFactor=0.79,
                nms=[0.5,0.6,0.7,0.7],
                threshold=[0.6,0.6,0.7],
                model_path=[pnet_path,rnet_path,onet_path],
                save_stage=True)
cap=cv2.VideoCapture(0)

while True:

    ret, frame=cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # frame=cv2.imread('images/M.jpg')
    # frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # ret=True
    if ret:
        stage_box=df.detect_face(frame)

        pImage=drawDectectBox(frame, stage_box[0], scores=stage_box[0][:, 4])
        if len(stage_box)>1:
            rImage=drawDectectBox(frame, stage_box[1], scores=stage_box[1][:, 4])
        else:
            rImage=pImage.copy()
        if len(stage_box)>2:
            oImage=drawDectectBox(frame, stage_box[2][:,:4], scores=stage_box[2][:, 4])
            oImage=drawLandMarks(oImage,stage_box[2][:,5:])
        else:
            oImage=pImage.copy()

        I= bigPicture(frame, pImage, rImage, oImage,320,426)[:, :, ::-1]
        cv2.imshow('myvideo',I)

    if cv2.waitKey(1)==ord('q'):break

cap.release()
cv2.destroyAllWindows()
