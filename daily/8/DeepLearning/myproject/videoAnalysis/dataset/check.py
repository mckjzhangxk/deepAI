from yolov3 import plot_one_box
import cv2
import matplotlib.pyplot as plt
if __name__ == '__main__':
    basepath='/home/zhangxk/abc/test/1/601'
    I=cv2.imread(basepath+'.jpg')
    H,W,_=I.shape

    with open(basepath+'.txt')as fs:

        for line in fs:
            sp=line.split()

            cx=float(sp[1])*W
            cy = float(sp[2])*H
            w = float(sp[3])*W
            h = float(sp[4])*H


            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            plot_one_box((x1,y1,x2,y2),I)
    plt.imshow(I[:,:,::-1])
    plt.show()
