import cv2
import glob
import os
from  yolov3 import plot_one_box
from utils import videoWriter,readVideo

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


    imgpath='/home/zxk/AI/faceswap-GAN/videoplayback.mp4'
    # imgpath = '/home/zxk/AI/faceswap-GAN/videoplayback1.webm'

    out='/home/zxk/AI/faceswap-GAN/1.avi'
    cap,info=readVideo(imgpath)
    writer = videoWriter(out, videoformat='XVID', scale=(info['width'],info['height']), fps=30)

    import tqdm
    start=2600
    # start=5400
    end=4440
    for i in tqdm.tqdm(range(start,start+25*30)):
        cap.set(1,i)
        _,frame=cap.read()
        # print(frame.shape)
        # print(frame)
        writer.write(frame)


    writer.release()
