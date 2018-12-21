import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.misc import imsave

# name='17--Ceremony/17_Ceremony_Ceremony_17_129.jpg'
# cnt=1
# line='941 301 22 29 1 0 1 0 0 0'
# filename=os.path.join(path,name)
#
# if os.path.exists(filename):
#     I=cv2.imread(filename)
#     I=I[:,:,::-1]
#     H,W,_=I.shape
#
#
#     for i in range(cnt):
#         splits=line.split(' ')
#         x1,y1,w,h=int(splits[0]),int(splits[1]),int(splits[2]),int(splits[3])
#         x2,y2=x1+w,y1+h
#
#         pad=int((0.8*np.random.rand()+0.2)*max(w,h))
#         subx1, suby1=max(x1-pad,0),max(y1-pad,0)
#         #临时的末端点
#         subx2,suby2=min(x2+pad,W),min(y2+pad,H)
#         subw,subh=subx2-subx1,suby2-suby1
#         l=max(subw,subh)
#         #最终末端点
#         subx2,suby2=min(subx1+l,W),min(suby1+l,H)
#
#         Iface=I[suby1:suby2,subx1:subx2,:]
#
#         boxX1=float(x1-subx1)/subw
#         boxX2 = float(x2 - subx1) / subw
#         boxY1=float(y1-suby1)/subh
#         boxY2 = float(y2 - suby1) / subh
#
#         I12=cv2.resize(Iface,(12,12),interpolation=cv2.INTER_AREA)
#         I24 =cv2.resize(Iface, (24, 24))
#         I48 =cv2.resize(Iface, (48, 48))
#
#         Y=np.array([0,1])
#         YBOX=np.array([boxX1,boxY1,boxX2,boxY2])


# cv2.imwrite('xx.jpg',I)

def onePicture(fs,path):
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

        tmp=[]
        for i in range(cnt):
            line = fs.readline()
            splits = line.split(' ')
            x1, y1, w, h = int(splits[0]), int(splits[1]), int(splits[2]), int(splits[3])
            x2, y2 = x1 + w, y1 + h


            pad = int((0.8 * np.random.rand() + 0.2) * max(w, h))
            subx1, suby1 = max(x1 - pad, 0), max(y1 - pad, 0)
            # 临时的末端点
            subx2, suby2 = min(x2 + pad, W), min(y2 + pad, H)
            subw, subh = subx2 - subx1, suby2 - suby1
            l = max(subw, subh)
            if subw==0 or subh==0:
                continue
            # 最终末端点
            subx2, suby2 = min(subx1 + l, W), min(suby1 + l, H)

            Iface = I[suby1:suby2, subx1:subx2, :]

            boxX1 = float(x1 - subx1) / subw
            boxX2 = float(x2 - subx1) / subw
            boxY1 = float(y1 - suby1) / subh
            boxY2 = float(y2 - suby1) / subh
            if Iface.size==0:
                print('error:',filename)
                print(x1, y1, w, h)
                continue

            #保存所有结果
            I12 = cv2.resize(Iface, (12, 12), interpolation=cv2.INTER_AREA)
            I24 = cv2.resize(Iface, (24, 24), interpolation=cv2.INTER_AREA)
            I48 = cv2.resize(Iface, (48, 48), interpolation=cv2.INTER_AREA)
            Y = np.array([0, 1])
            YBOX = np.array([boxX1, boxY1, boxX2, boxY2])

            RetI12.append(I12)
            RetI24.append(I24)
            RetI48.append(I48)
            RetY.append(Y)
            RetYBOX.append(YBOX)

            #一个正例对应一个负例,+-例图片大小应该一样.都是lxl的图片,并且不想交叉
            tmp.append((x1, y1, x2, y2,l))
            #负例

        numExample=len(tmp)
        for i in range(numExample):
            nx1,ny1,nx2,ny2,Ineg=sampleNeg(I,tmp,i)
            #这一步是为了生成的新负例子,不和已产生的例子相交
            tmp.append((nx1,ny1,nx2,ny2,-1))
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

    return RetI12,RetI24,RetI48,RetY,RetYBOX

def sampleNeg(I,tmp,i):
    def intersect(x1,y1,x2,y2):
        for bx1,by1,bx2,by2,_ in tmp:
            cx1,cy1=max(x1,bx1),  max(y1,by1)
            cx2,cy2=min(x2, bx2), min(y2, by2)

            if cx1<=cx2 and cy1<=cy2:return True
        return False
    H,W,_=I.shape
    size=tmp[i][4]

    while True:
        x1=np.random.randint(0,W-size)
        y1=np.random.randint(0, H-size)
        x2=x1+size
        y2=y1+size

        if intersect(x1,y1,x2,y2)==False:
            return (x1,y1,x2,y2,I[y1:y2,x1:x2,:])
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
    path='/home/zhangxk/下载/WIDER_train/images'
    path_sample='/home/zhangxk/下载/samples'
    path_box='/home/zhangxk/下载/wider_face_split/wider_face_train_bbx_gt.txt'

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
    fs.close()
