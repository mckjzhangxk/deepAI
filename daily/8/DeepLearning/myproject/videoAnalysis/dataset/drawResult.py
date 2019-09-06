import cv2
import glob
import os
from  yolov3 import plot_one_box
from utils import videoWriter

def listlabel(path):
    return glob.glob(path)
def parse(filename,imgpath):
    with open(filename) as fs:
        img_relative_path=fs.readline().strip()
        cnt=int(fs.readline().strip())

        I=cv2.imread(os.path.join(imgpath,img_relative_path))

        for i in range(cnt):
            lines=fs.readline().strip().split(' ')

            x1,y1=int(lines[0]),int(lines[1])
            x2,y2=x1+int(lines[2]),y1+int(lines[3])
            plot_one_box([x1,y1,x2,y2],I,None);
        return I
if __name__ == '__main__':
    filelist1=listlabel('/home/zhangxk/widerface/java_arcface_result/*/*.txt')
    filelist1=sorted(filelist1)

    filelist2=listlabel('/home/zhangxk/widerface/yolo_result/*/*.txt')
    filelist2=sorted(filelist2)

    imgpath='/home/zhangxk/AIProject/数据集与模型/WINDER_Face/WIDER_val/images'
    scale=(1024,768)
    writer = videoWriter('output.avi', videoformat='XVID', scale=(scale[0]*2,scale[1]), fps=1)

    import tqdm
    for filename1,filename2 in tqdm.tqdm(zip(filelist1,filelist2)):
        I1=parse(filename1,imgpath)
        I1=cv2.resize(I1,scale)

        I2=parse(filename2,imgpath)
        I2=cv2.resize(I2,scale)

        import numpy as np
        I=np.concatenate((I1,I2),1)
        writer.write(I)
        del I
    writer.release()
