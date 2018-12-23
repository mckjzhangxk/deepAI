import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


def filter(I,faceid,line):
    I=I.copy()
    splits=line.split(' ')


    x1, y1, w, h = int(splits[0]), int(splits[1]), int(splits[2]), int(splits[3])
    x2, y2 = x1 + w, y1 + h
    blur, expression, illumination, invalid, occlusion, pose=int(splits[4]),int(splits[5]),int(splits[6]),int(splits[7]),int(splits[8]),int(splits[9])



    # if _invalid==invalid:
    #     cv2.imwrite('hello.jpg',I[y1:y2,x1:x2])
    #     print('ss')
    if invalid==0 and occlusion==0 and w>=12 and h>=12:
        cv2.imwrite('hello.jpg', I)
        # invalid='invalid' if invalid==1 else ''
        # occlusion='occ' if occlusion>0 else ''
        # illumination='E' if illumination>0 else 'N'
        # pose='H' if pose>0 else 'L'
        cv2.imwrite('%d.jpg' % (faceid), I[y1:y2, x1: x2])
        cv2.rectangle(I, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.imwrite('hello%d.jpg' % (faceid), I)

        return x1,y1,x2,y2,w,h
    else:return None
def testMyPicture(I,I12,I24,I48):
    cv2.imwrite('test/I.jpg' , I)
    for k in range(len(I12)):
        cv2.imwrite('test/I12_%d.jpg'%k,I12[k])
        cv2.imwrite('test/I24_%d.jpg' % k, I24[k])
        cv2.imwrite('test/I48_%d.jpg' % k, I48[k])

def onePicture(fs,path):
    def isNotSquare(X):return X.shape[0]!=X.shape[1]
    name=fs.readline()
    if name=='':return None
    name=name.replace('\n','')
    cnt=int(fs.readline())

    filename = os.path.join(path, name)


    RetI12,RetI24,RetI48=[],[],[]
    RetY,RetYBOX=[],[]

    if os.path.exists(filename):
        I = cv2.imread(filename)
        I = I[:, :, ::-1]
        H, W, _ = I.shape

        tmp=[]#save() face position,then using it to crop non-face region later
        for i in range(cnt):
            line = fs.readline()
            fret=filter(I,i ,line)
            if fret==None:continue
            x1, y1, x2, y2, w, h=fret

            #预留出4%-15%的margin
            pad = int(0.5 * min(w, h))
            print(pad)
            sx1, sy1 = max(x1 - pad, 0), max(y1 - pad, 0)
            # 临时的末端点
            sx2, sy2 = min(x2 + pad, W), min(y2 + pad, H)
            sw, sh = sx2 - sx1, sy2 - sy1
            l = max(sw, sh)

            # 最终末端点
            sx2, sy2 = min(sx1 + l, W), min(sy1 + l, H)

            Iface=np.zeros((l,l,3),dtype=np.uint8)
            Iface[0:sy2-sy1,0:sx2-sx1]=I[sy1:sy2,sx1:sx2]

            cv2.imwrite('H%d.jpg' % (i), Iface)

            boxX1 = float(x1 - sx1) / sw
            boxX2 = float(x2 - sx1) / sw
            boxY1 = float(y1 - sy1) / sh
            boxY2 = float(y2 - sy1) / sh

            #保存所有结果
            I12 = cv2.resize(Iface, (12, 12), interpolation=cv2.INTER_AREA)
            I24 = cv2.resize(Iface, (24, 24), interpolation=cv2.INTER_AREA)
            I48 = cv2.resize(Iface, (48, 48), interpolation=cv2.INTER_AREA)
            Y = np.array([0, 1])
            YBOX = np.array([boxX1, boxY1, boxX2, boxY2])
            print(YBOX)
            RetI12.append(I12)
            RetI24.append(I24)
            RetI48.append(I48)
            RetY.append(Y)
            RetYBOX.append(YBOX)


            #一个正例对应一个负例,+-例图片大小应该一样.都是lxl的图片,并且不想交叉
            tmp.append((x1, y1, x2, y2))
            #负例

        numExample=len(tmp)
        for i in range(numExample):
            r=sampleNeg(I,tmp,i)
            if r is None:continue
            nx1,ny1,nx2,ny2,Ineg=r
            #这一步是为了生成的新负例子,不和已产生的例子相交
            tmp.append((nx1,ny1,nx2,ny2))
            I12 = cv2.resize(Ineg, (12, 12), interpolation=cv2.INTER_AREA)
            I24 = cv2.resize(Ineg, (24, 24), interpolation=cv2.INTER_AREA)
            I48 = cv2.resize(Ineg, (48, 48), interpolation=cv2.INTER_AREA)
            Y = np.array([1, 0])
            YBOX = np.array([0, 0, 0, 0])#dont care

            RetI12.append(I12)
            RetI24.append(I24)
            RetI48.append(I48)
            RetY.append(Y)
            RetYBOX.append(YBOX)
    # testMyPicture(I,RetI12,RetI24,RetI48)
    return RetI12,RetI24,RetI48,RetY,RetYBOX

def sampleNeg(I,tmp,i,maxTry=10):
    def intersect(x1,y1,x2,y2):
        for bx1,by1,bx2,by2 in tmp:
            cx1,cy1=max(x1,bx1),  max(y1,by1)
            cx2,cy2=min(x2, bx2), min(y2, by2)

            if cx1<=cx2 and cy1<=cy2:return True
        return False
    H,W,_=I.shape
    w,h=tmp[i][2]-tmp[i][0],tmp[i][3]-tmp[i][1]

    if W==w or H==h:return None

    while maxTry>0:
        x1=np.random.randint(0,W-w)
        y1=np.random.randint(0, H-h)
        x2=x1+w
        y2=y1+h

        if intersect(x1,y1,x2,y2)==False:
            return (x1,y1,x2,y2,I[y1:y2,x1:x2,:])
        maxTry-=1
    return None
def outputSample(path,round,RetI12,RetI24,RetI48,RetY,RetYBOX):
    filename1 = os.path.join(path,'X12_%d'%round)
    filename2 = os.path.join(path,'X24_%d.npy' % round)
    filename3 = os.path.join(path,'X48_%d.npy' % round)
    filename4 = os.path.join(path,'Y_%d.npy' % round)
    filename5 = os.path.join(path,'YBOX_%d.npy' % round)

    np.save(filename1, np.array(RetI12))
    np.save(filename2, np.array(RetI24))
    np.save(filename3, np.array(RetI48))
    np.save(filename4, np.array(RetY))
    np.save(filename5, np.array(RetYBOX))

if __name__ == '__main__':
    N=1000
    path='/home/zxk/AI/data/widerface/WIDER_train/images'
    path_sample='/home/zxk/AI/data/widerface/WIDER_train/samples'
    path_box='/home/zxk/AI/data/widerface/wider_face_split/wider_face_train_bbx_gt.txt'

    fs=open(path_box)
    RetI12,RetI24,RetI48,RetY,RetYBOX=[],[],[],[],[]
    cnt=0
    while True:
        ret = onePicture(fs, path)
        if ret is None:break
        RetI12.extend(ret[0])
        RetI24.extend(ret[1])
        RetI48.extend(ret[2])
        RetY.extend(ret[3])
        RetYBOX.extend(ret[4])
        cnt+=1

        if cnt%N==0:
            outputSample(path_sample,cnt//N,RetI12,RetI24,RetI48,RetY,RetYBOX)
            print(cnt // N, len(RetI12))
            RetI12, RetI24, RetI48, RetY, RetYBOX = [], [], [], [], []
    outputSample(path_sample, (cnt // N)+1, RetI12, RetI24, RetI48, RetY, RetYBOX)
    fs.close()
