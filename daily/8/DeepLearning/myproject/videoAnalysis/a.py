from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from utils import prewhiten
import cv2
import numpy as np

if __name__ == '__main__':
    # model=InceptionResnetV1(pretrained='casia-webface').eval()
    # I=cv2.imread('/home/zxk/PycharmProjects/deepAI1/daily/8/mysite/webapp/facedb1/4.png')
    # I = I[:, :, ::-1]
    # I=cv2.resize(I,(160,160))
    # I = I[None]
    # I=np.transpose(I,(0,3,1,2))
    # I = prewhiten(I)
    #
    # faceid=model(I).data.cpu().numpy()
    # print(np.linalg.norm(faceid[0]))
    x=[(1,'a'),(2,'b')]

    x=list(zip(*x))
    print(x)