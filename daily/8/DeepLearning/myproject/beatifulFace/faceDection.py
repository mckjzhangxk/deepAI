__author__ = 'mathai'
import dlib
import cv2
import numpy as np
from collections import OrderedDict

modelPath='data/shape_predictor_68_face_landmarks.dat'
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor(modelPath)


FACIAL_LANDMARKS_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 35)),
	("jaw", (0, 17))
])

def two2Four(pts):
    ret=[]

    ret.append(pts[0])
    ret.append(np.array([pts[1][0],pts[0][1]]))
    ret.append(pts[1])
    ret.append(np.array([pts[0][0],pts[1][1]]))
    return np.array(ret)
def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)
def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords

def detectFace(imgpath):

    img=cv2.imread(imgpath)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)



    rects=detector(gray,1)
    shape = shape_to_np(predictor(gray, rects[0]))

    return cv2.cvtColor(img,cv2.COLOR_BGR2RGB),rect_to_bb(rects[0]),shape,

def getRightEyeBow(shape):
    ii=FACIAL_LANDMARKS_IDXS['right_eyebrow']
    pts=shape[ii[0]:ii[1]]

    xy_min=np.min(pts,axis=0)
    xy_max=np.max(pts,axis=0)
    return pts,two2Four(np.array([xy_min,xy_max]))
def getLeftEyeBow(shape):
    ii=FACIAL_LANDMARKS_IDXS['left_eyebrow']
    pts=shape[ii[0]:ii[1]]

    xy_min=np.min(pts,axis=0)
    xy_max=np.max(pts,axis=0)
    return pts,two2Four(np.array([xy_min,xy_max]))

def getJaw(shape):
    ii=FACIAL_LANDMARKS_IDXS['jaw']
    ii=[3,4,5,11,12,13]
    pts=shape[ii]
    return ii,pts
    
if __name__ == '__main__':
    imgpath=r'data/face1.jpeg'
    modelPath='data/shape_predictor_68_face_landmarks.dat'

    img=cv2.imread(imgpath)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    detector=dlib.get_frontal_face_detector()
    predictor=dlib.shape_predictor(modelPath)


    rects=detector(gray,1)



    for rect in rects:
        x,y,w,h=rect_to_bb(rect)
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        shape = shape_to_np(predictor(gray, rect))
        for i in range(len(shape)):
            cv2.circle(img,tuple(shape[i].tolist()),3,(255,0,0),1)
            cv2.putText(img, str(i+1), (shape[i][0] - 10, shape[i][1] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

    cv2.imshow('Output',img)
    cv2.waitKey(0)