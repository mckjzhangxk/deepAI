from Configure import WIDER_ANNOTION,WIDER_TRAINSET
import os
import numpy as np
import cv2
'''
返回一个dict
    key:图片名:
    value:np.array([N',4])标注的人脸
    这里过滤了过小人脸(size<12)
    
    total num of images 12880
    total num of faces 94484
'''
def get_WIDER_Set():
    ret=dict()

    fs = open(WIDER_ANNOTION, 'r')
    lines=fs.readlines()
    cnt=len(lines)
    numOFImages=0
    numOFFace=0
    SIZE=12
    idx=0


    while (idx < cnt):
        name = lines[idx].strip('\n')
        imagepath = os.path.join(WIDER_TRAINSET, name)
        I = cv2.imread(imagepath)
        H, W, _ = I.shape

        # 获得人脸数目
        idx += 1
        facenum = int(lines[idx].strip('\n'))

        face_coordnate=[]
        for n in range(facenum):
            idx+=1
            splits = lines[idx].split(' ')
            x1, y1, w, h = float(splits[0]), float(splits[1]), float(splits[2]), float(splits[3])
            x2, y2 = x1 + w, y1 + h
            if (x1 + SIZE < x2 and x1 >= 0 and x2 <= W and y1 + SIZE < y2 and y1 >= 0 and y2 <= H):
                face_coordnate.append((x1,y1,x2,y2))
        numOFImages += 1
        numOFFace+=len(face_coordnate)
        #跳到下一张图片
        idx+=1
        ret[imagepath]=np.array(face_coordnate)
    fs.close()

    print('total num of images %d' % numOFImages)
    print('total num of faces %d' % numOFFace)
    return ret


def get_WIDER_Set_ImagePath():
    ret=[]
    fs = open(WIDER_ANNOTION, 'r')
    lines = fs.readlines()
    cnt = len(lines)

    idx = 0

    while (idx < cnt):
        name = lines[idx].strip('\n')
        imagepath = os.path.join(WIDER_TRAINSET, name)

        # 获得人脸数目
        idx += 1
        facenum = int(lines[idx].strip('\n'))
        idx =idx+ facenum+1
        if os.path.exists(imagepath):
            ret.append(imagepath)
    fs.close()
    return ret