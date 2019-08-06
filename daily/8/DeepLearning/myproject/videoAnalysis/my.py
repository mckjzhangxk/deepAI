# from facenet_pytorch import MTCNN,InceptionResnetV1
import cv2
from PIL import Image
from mtcnn_pytorch import MTCNN
from utils import AlignFace
if __name__ == '__main__':
    # img=Image.open('/home/zhangxk/projects/deepAI/daily/8/DeepLearning/myproject/videoAnalysis/data/20190605000079.jpg','r')
    img=Image.open('/home/zhangxk/projects/deepAI/daily/8/DeepLearning/myproject/videoAnalysis/data/2019-08-05 14-34-07屏幕截图.png')
    mtcnn=MTCNN('cpu')
    boxes, landmarks = mtcnn.detect_faces(img)
    align=AlignFace()

    for i in range(len(boxes)):
        I=align.align(img,boxes[i],landmarks[i],20,flag=1)
        I.show('my')


