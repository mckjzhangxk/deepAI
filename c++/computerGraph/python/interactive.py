import cv2
import numpy as np

path=[]
SCREEN_SIZE=600
img=np.zeros((SCREEN_SIZE,SCREEN_SIZE,3),np.uint8)
title='mycanvas'

def click_event(event,x,y,flags,param):
    global path
    global img
    if event==cv2.EVENT_LBUTTONDOWN:
        path.append([x,y])
        cv2.circle(img,(x,y),2,(255,0,0))
        if len(path)>1:
            cv2.polylines(img,[np.array(path)],False,(255,0,0),thickness=2)
        cv2.imshow(title,img)
    elif event==cv2.EVENT_RBUTTONDOWN:
        print(path)
        path=[]
        img=np.zeros((SCREEN_SIZE,SCREEN_SIZE,3),np.uint8)
        cv2.imshow(title,img)
if __name__ == "__main__":
    cv2.imshow(title,img)
    cv2.setMouseCallback(title,click_event)    
    cv2.waitKey(0)