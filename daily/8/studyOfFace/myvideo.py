from mylib.detect.detect_face import create_mtcnn,p_stage,r_stage,o_stage,drawLandMarks,drawDectectBox
import tensorflow as tf
import numpy as np
import cv2

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

sess=tf.Session()
pnet,rnet,onet=create_mtcnn(sess)
#saver=tf.train.Saver()
#saver.restore(sess,'/home/zxk/PycharmProjects/deepAI/daily/8/studyOfFace/logs/models/facedect.ckpt-37')
cap=cv2.VideoCapture(0)

while True:

    ret, frame=cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # frame=cv2.imread('images/M.jpg')
    # frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # ret=True
    if ret:
        boxes, out = p_stage(frame, pnet=pnet, minsize=50, factor=0.709, t=0.6, debug=False)
        pImage=drawDectectBox(frame, boxes, scores=boxes[:, 4])


        boxes, out=r_stage(frame, boxes, out, rnet=rnet, t=0.6, debug=False)
        rImage=drawDectectBox(frame, boxes, scores=boxes[:, 4])

        boxes, landmark, score=o_stage(frame, boxes, out, onet=onet, t=0.7, debug=False)
        if len(boxes)>0:
            oImage=drawLandMarks(drawDectectBox(frame, boxes, scores=score), landmark)
        else:
            oImage=frame

        I= bigPicture(frame, pImage, rImage, oImage,320,426)[:, :, ::-1]
        cv2.imshow('myvideo',I)

    if cv2.waitKey(1)==ord('q'):break

cap.release()
cv2.destroyAllWindows()
