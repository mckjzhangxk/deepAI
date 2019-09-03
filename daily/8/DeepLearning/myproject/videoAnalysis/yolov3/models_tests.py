import torch
from yolov3.yoloUtils import CCPD_YOLO_Detector
from yolov3.utils.utils import plot_one_box
import cv2


def createModel():
   device=torch.device('cpu')
   CCPD_YOLO_Detector(device=device)

def detect():
   import  glob
   import  os

   imglist=glob.glob('../data/yolo/*.jpg')
   device=torch.device('cpu')
   detector=CCPD_YOLO_Detector(device=device)

   batchImage=[cv2.imread(imgfile) for imgfile in imglist]
   results = detector.predict(batchImage)

   for I,imgfile,result in zip(batchImage,imglist,results):

      for x1,y1,x2,y2,conf,cls in result:
         plot_one_box((x1,y1,x2,y2),I,label='%.2f'%conf,color=(0,255,0))
      outfile=os.path.join('../data/output',os.path.basename(imgfile))
      cv2.imwrite(outfile,I)

if __name__ == '__main__':
   # createModel()
   detect()