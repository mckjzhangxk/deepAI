# from facenet_pytorch import MTCNN,InceptionResnetV1
import cv2
from PIL import Image
from mtcnn_pytorch import MTCNN
from utils import AlignFace
from utils.alignUtils import get_reference_facial_points
if __name__ == '__main__':
    img=Image.open('/home/zhangxk/projects/deepAI/daily/8/DeepLearning/myproject/videoAnalysis/data/20190605000079.jpg','r')
    img=Image.open('/home/zhangxk/projects/deepAI/daily/8/DeepLearning/myproject/videoAnalysis/data/spiderman/神盾局局长.png')
    mtcnn=MTCNN('cpu')
    boxes, landmarks = mtcnn.detect_faces(img)
    align=AlignFace((160,160))

    print(landmarks[0].shape)
    for i in range(len(boxes)):
        I=align.align(img,boxes[i],landmarks[i],40,flag=1)
        I.show('my')
        print(I.size)




