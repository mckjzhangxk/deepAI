from facenet_pytorch import MTCNN,InceptionResnetV1
import cv2
from PIL import Image

if __name__ == '__main__':
    mtcnn=MTCNN(
        image_size=160,
        margin=0,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        prewhiten=True,
        select_largest=True,#True,boxes按照面积的大小降序排列
        keep_all=True,
        device=None
    )

    img = Image.open('data/multiface.jpg')
    img=cv2.imread('data/multiface.jpg')

    faces=mtcnn(img,save_path='11')
    model=InceptionResnetV1(pretrained='casia-webface').eval()
    a=model(faces)
    print(a.shape)

