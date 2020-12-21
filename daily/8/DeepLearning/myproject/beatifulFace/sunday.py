import numpy as np
import cv2
from faceDection import detectFace,getRightEyeBow,getLeftEyeBow
from blend import blend,process

def drawLine(img,pts,color=(255,0,0)):
    for i in range(1,len(pts)):
        cv2.line(img,tuple(pts[i-1]),tuple(pts[i]),color,1)
    cv2.line(img,tuple(pts[0]),tuple(pts[-1]),color,1)
    return img


def normalize(x):
    return x/np.linalg.norm(x)

def meatureDistanceAlongAxis(pts,v):
    ds=pts.dot(v)
    return np.max(ds)-np.min(ds)

def getBoundRect(pts):
    v1=normalize((pts[4]-pts[0]))
    v2=np.array([v1[1],-v1[0]])

    W=meatureDistanceAlongAxis(pts,v1)
    H=meatureDistanceAlongAxis(pts,v2)

    assert np.allclose(W,np.linalg.norm(pts[0]-pts[4]))


    pA=pts[0]+H*v2
    pB=pA+W*v1;
    pC=pts[0]+W*v1;
    pD=pts[0]

    return np.vstack((pA,pB,pC,pD)).astype('int')


def getMask(image,pts):
    '''
    image:gray scale
    pts:眉毛的坐标点
    '''
    minxy=pts.min(axis=0)
    maxxy=pts.max(axis=0)
    
    X,Y=np.meshgrid(range(minxy[0],maxxy[0]),range(minxy[1],maxxy[1]),indexing='ij')
    X=X.ravel()
    Y=Y.ravel()
    # convex_hull=[]
    # for x,y in zip(X,Y):
    #     if cv2.pointPolygonTest(pts,(x,y),False)>0:
    #         convex_hull.append((x,y))
    # convex_hull=np.array(convex_hull)    #坐标
    
    ROI=image[Y,X]  #坐标值
    EYE_BROW_MASK=(ROI<np.mean(ROI)+0.2*np.std(ROI))  #眉毛
    
    Z=np.zeros((image.shape[:2])).astype('uint8')
    Z[Y,X]=EYE_BROW_MASK

    kernel=np.ones((5,5),'uint8')
    Z = cv2.morphologyEx(Z, cv2.MORPH_OPEN, kernel)

    kernel= cv2.getStructuringElement(cv2.MORPH_CROSS,(9,9))
    Z=cv2.dilate(Z,kernel)
    # Z=cv2.morphologyEx(Z, cv2.MORPH_CLOSE, kernel)

    return Z
def rightEyebrowFeature(img,feature_pts):
    '''
        image :RGB图片
        feature_pts:5个眉毛坐标

        返回:
        align_box:array(int),4x2,轴对称的体格顶点
        mask:array(uint8), img.shape[:2],眉毛的mask
    '''
    
    align_box=getBoundRect(feature_pts)
    img_gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    mask=getMask(img_gray,feature_pts)

    return align_box,mask
def learn(M1,M2):

    '''
    TM1=M2
    :param M1:
    :param M2:
    :return:
    '''
    u1=np.mean(M1,axis=0)
    u2=np.mean(M2,axis=0)

    M1=M1-u1
    M2=M2-u2

    A=np.zeros((len(M1)*2,4),'float')
    b=M2.ravel()

    A[::2,:2]=M1
    A[1::2,2:]=M1


    coffe,residuals,rank,s=np.linalg.lstsq(A,b)

    T=np.reshape(coffe,(2,2))
    t=-T.dot(u1)+u2

    return T,t

def transformMatrix(T,t):
    return np.hstack((T,t[:,np.newaxis]))
def transformApply(pts,T,t):
    '''
    把pts 的的点 转换T，平移的坐标转换，返回int 类型的变换后的坐标
    一行pts表示一个点，返回的表示也是一样

    T：float(2,2)针对一列，一列表示一个点的先行变化
    t:float array(2),针对一列
    '''
    AA=T.dot(pts.T)+t[:,np.newaxis]
    return np.int32(AA.T);

def readFace(facepath):
    result={}
    img,bbox,feature_pts=detectFace(facepath)
    result['feature']=feature_pts
    result['image']=img
    result['w']=img.shape[1]
    result['h']=img.shape[0]
    result['bbox']=bbox

    result['right_eyebrow']={}
    right_eyebrow_pts,right_eyebrow_rect=getRightEyeBow(feature_pts)
    rects,mask=rightEyebrowFeature(img,right_eyebrow_pts)
    result['right_eyebrow']['feature']=right_eyebrow_pts
    result['right_eyebrow']['bbox']=rects
    result['right_eyebrow']['mask']=mask


    result['left_eyebrow']={}
    left_eyebrow_pts,left_eyebrow_rect=getLeftEyeBow(feature_pts)
    rects,mask=rightEyebrowFeature(img,left_eyebrow_pts)

    result['left_eyebrow']['feature']=left_eyebrow_pts
    result['left_eyebrow']['bbox']=rects
    result['left_eyebrow']['mask']=mask

    return result


def  ABC(a,b,c):
    H,W,_=a.shape
    
    bW=int(b.shape[1]*H/b.shape[0])
    b=cv2.resize(b,(bW,H))

    Z=np.zeros((H,2*W+bW,3),dtype=np.uint8)

    Z[:,0:W]=a
    Z[:,W:W+bW]=b
    Z[:,W+bW:]=c
    return Z
def  replaceEyebrow(face1,face2):
    
    target=face1['image'].copy()
    target_H,target_W=target.shape[:2]

    for _key in ['right_eyebrow','left_eyebrow']:
        bbox1=face1[_key]['bbox']
        bbox2=face2[_key]['bbox']
        T,t=learn(bbox2,bbox1)

        mask1=face1[_key]['mask'][:,:,np.newaxis]
        mask2=face2[_key]['mask'][:,:,np.newaxis]

        src=cv2.warpAffine(face2['image'],transformMatrix(T,t),dsize=(target_W,target_H))
        mask2=cv2.warpAffine(mask2,transformMatrix(T,t),dsize=(target_W,target_H))
        mask=cv2.bitwise_or(mask1,mask2)

        source=cv2.filter2D(np.float32(src),-1,np.float32([
            [0,-1,0],[-1,4,-1],[0,-1,-0]
        ]))
        target=process(source, target, mask)
    return target

def thinFace(face1,s):
    pass
if __name__ == '__main__':
    face1=readFace('data/face3.jpeg')
    face2=readFace('data/face5.jpeg')

    ##############眉毛测试 bbox mask#######################################################
    #画2个人脸
    # img=drawLine(face1['image'],face1['right_eyebrow']['bbox'],(0,255,0))
    # img=drawLine(face1['image'],face1['left_eyebrow']['bbox'],(0,255,0))
    # imgtest=img*(1-face1['right_eyebrow']['mask'][:,:,np.newaxis])
    # imgtest=imgtest*(1-face1['left_eyebrow']['mask'][:,:,np.newaxis])
    
    # cv2.imshow('face1 bbox mask test',imgtest)
    #####################################################################


    img_result=replaceEyebrow(face1,face2)
    img=ABC(face1['image'],face2['image'],img_result)
    # cv2.imshow('src',cv2.cvtColor(face2['image'],cv2.COLOR_RGB2BGR))
    cv2.imshow('demo',cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
    # cv2.imshow('original',cv2.cvtColor(face1['image'],cv2.COLOR_RGB2BGR))

   
    while True:
        key=cv2.waitKey(0)

        if key==ord('a'):
            kernel=np.ones((7,7),'int')
            # mask=cv2.dilate(mask,kernel)
            mask=cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # mask=mask[:,:,np.newaxis]
            # img=src*mask+target*(1-mask)

            # cv2.imshow('paste result',cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
        if key==ord('q'):
            break
     
