import numpy as np
import matplotlib.pyplot as plt
import cv2


def _bilinear(x,y,f):
    x1,y1=np.floor(x).astype(np.int),np.floor(y).astype(np.int)
    x2,y2=np.ceil(x).astype(np.int), np.ceil(y).astype(np.int)
    A00,A01,A10,A11=f[y1,x1],f[y1,x2],f[y2,x1],f[y2,x2]

    if f.ndim==3:
        x,y=np.expand_dims(x,1),np.expand_dims(y,1)
        x1, y1 = np.expand_dims(x1, 1), np.expand_dims(y1, 1)
        x2, y2 = np.expand_dims(x2, 1), np.expand_dims(y2, 1)
    eps=1e-6
    f00 = ((x2 - x) * (y2 - y) + eps) / ((x2 - x1) * (y2 - y1) + eps)*A00
    f01 = ((x - x1) * (y2 - y) + eps) / ((x2 - x1) * (y2 - y1) + eps)*A01
    f10 = ((x2 - x) * (y - y1) + eps) / ((x2 - x1) * (y2 - y1) + eps)*A10
    f11 = ((x - x1) * (y - y1) + eps) / ((x2 - x1) * (y2 - y1) + eps)*A11

    res=f00+f01+f10+f11
    res[np.where(x1 == x2)[0]] /= 2
    res[np.where(y1 == y2)[0]] /= 2
    return res


def _tik(oldL,newL):
    space=oldL/newL
    x=np.arange(1,newL+1)*space-(space+1)/2

    return x
def _bi_resize(I,newShape):
    H,W=I.shape[0],I.shape[1]
    nw,nh=newShape

    xrange=_tik(W,nw)+1
    yrange=_tik(H,nh)+1
    X,Y=np.meshgrid(xrange,yrange)

    if I.ndim==3:
        f=np.pad(I,((1,1),(1,1),(0,0)),'symmetric')
    else:
        f = np.pad(I, ((1, 1), (1, 1)), 'symmetric')
    Inew=_bilinear(X.ravel(),Y.ravel(),f).astype(I.dtype)
    return np.reshape(Inew,(nh,nw,*I.shape[2:]))

# I=np.random.randint(0,255,(32,32,3))
scale=1.6
I=cv2.imread('data/demo_data/dog.jpg')

# I=cv2.cvtColor(I,cv2.COLOR_BGR2RGB)
H,W,_=I.shape

nH,nW=int(scale*H),int(scale*W)
Inew=_bi_resize(I,(80,50))
print(Inew.shape)
cv2.imshow('xx',Inew)
cv2.waitKey()