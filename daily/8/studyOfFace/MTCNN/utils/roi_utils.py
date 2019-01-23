import numpy as np
import numpy.random as npr
import cv2
def IoU(box, boxes):
    """Compute IoU between detect box and gt boxes

    Parameters:
    ----------
    box: numpy array , shape (5, ): x1, y1, x2, y2, score
        predicted boxes
    boxes: numpy array, shape (n, 4): x1, y1, x2, y2
        input ground truth boxes

    Returns:
    -------
    ovr: numpy.array, shape (n, )
        IoU
    """
    if isinstance(box,list):
        box=np.array(box)
    if isinstance(boxes,list):
        boxes=np.array(boxes)
    if boxes.ndim==1:
        boxes=np.expand_dims(boxes,0)

    box_area = (box[2] - box[0]) * (box[3] - box[1])
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3]- boxes[:, 1])
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)

    inter = w * h
    ovr = inter / (box_area + area - inter)
    return ovr

'''
把人脸坐标,映射到目标图片的坐标系中,返回
人脸左上角相对于目标左上角的偏移
人脸右上角相对于目标右上角的偏移

传入的都是绝对坐标

返回[regx1,regy1,regx2,regy2],
'''
def GetRegressBox(faceCoord,imageCoord):
    nx1,ny1,nx2,ny2=imageCoord
    w,h=float(nx2-nx1),float(ny2-ny1)

    x1,y1,x2,y2=faceCoord
    return [
        (x1-nx1)/w,
        (y1-ny1)/h,
        (x2-nx2)/w,
        (y2-ny2)/h
    ]


'''
判断一个box是否是一个合法的区域
'''
def validRegion(box,W,H):
    x1,y1,x2,y2=box[0],box[1],box[2],box[3]
    return x1>=0 and y1>=0 and x2<=W and y2<=H and x1<x2 and y1<y2

def validLandmark(landmark):
    return np.min(landmark)>0 and np.max(landmark)<1
class ImageTransform():
    '''
    
    给出一个boxm[x1,y1,x2,y2],[x1,x2),[y1,y2)
    以及边境[0,W) [0,h],随机平移区域
    生成boxsize~(0.8min,1.25max)的新box
    新box中心相对于传入box平移了dx~(-0.2w,0.2w),dy~(-0.2h,0.2h)
    
    最后会进行区域有效性验证,不合格会继续...
    返回[nx1,ny1,nx2,ny2]
    '''
    def shift(self,box,W,H,deep=0):
        if deep>=10:return [-1,-1,-1,-1]
        w,h=box[2]-box[0],box[3]-box[1]
        cenx,centy=box[0]+w//2,box[1]+h//2
        lmin,lmax=min(w,h),max(w,h)

        deltex=npr.randint(int(-0.2*w),int(0.2*w))
        deltey=npr.randint(int(-0.2*h),int(0.2*h))
        boxsize=npr.randint(int(0.8*lmin),int(1.25*lmax))

        nx1,ny1=max(cenx+deltex-boxsize//2,0),max(centy+deltey-boxsize//2,0)
        nx2,ny2=nx1+boxsize,ny1+boxsize

        retbox=[nx1,ny1,nx2,ny2]
        if validRegion(retbox,W,H):
            return retbox
        else:
            return self.shift(box,W,H,deep+1)

    '''
    传入一张图片,水平做翻转,返回翻转图片和landmark
    [H,W,3],landmark[5,2],传入的landmark已经norm
    '''
    def flip(self,I,landmark=None):
        I=I.copy()
        landmark=landmark.copy()
        if landmark.ndim==1:
            landmark=np.reshape(landmark,(-1,2))

        landmark[:,0]=1-landmark[:,0]
        landmark[[0,1]]=landmark[[1,0]]
        landmark[[3,4]]=landmark[[4,3]]

        I=np.flip(I,1)
        return I,landmark

    '''
    b:[N,2]
    box:x1,y1,x2,y2
    把b 投影到box上,返回以boxes左上角为0点的坐标
    ret[N,2]
    '''
    def project(self,b,boxes,to=True):
        b=b.copy()
        if isinstance(b,list):
            b=np.array(b)
        if b.ndim==1:
            b=np.reshape(b,(-1,2))
        x1,y1=boxes[0],boxes[1]
        b=b.copy()
        if to:
            b[:,0]=b[:,0]-x1
            b[:,1]=b[:,1]-y1
        else:
            b[:,0]=b[:,0]+x1
            b[:,1]=b[:,1]+y1

        return b

    '''

    facebox:人脸的绝对坐标(4,)x1,y1,x2,y2
    landmark:五官坐标,要有10个元素
    返回:[5,2]的np array
    '''

    def projectAndNorm(self,landmark,facebox):
        if isinstance(landmark, list):
            landmark = np.array(landmark)
        landmark = landmark.copy()
        if landmark.ndim != 2:
            landmark = np.reshape(landmark, (-1, 2))
        w = facebox[2] - facebox[0]
        h = facebox[3] - facebox[1]
        x1, y1 = facebox[0], facebox[1]

        landmark[:, 0] = (landmark[:, 0] - x1) / w
        landmark[:, 1] = (landmark[:, 1] - y1) / h
        return landmark
    '''
        输入一张图片,转angle角度,+表示逆时针
        -是顺时针
        landmark:(N,2)
        是相对于I坐标系的坐标
    '''
    def rotate(self,I,landmark,angle):
        I=I.copy()
        landmark=landmark.copy()
        if isinstance(landmark, list):
            landmark = np.array(landmark)
        if landmark.ndim == 1:
            landmark=np.reshape(landmark,(-1,2))

        H,W,_=I.shape
        cenx,ceny=W//2,H//2
        '''
        rot_mat,[2,3]
            [cos(-angle) -sin(-angle) 1-cos(-angle)*ux+sin(-angle)*uy 
             sin(-angle)  cos(-angle) -sin(-angle)*ux+1-cos(-angle)*uy
            ]
        '''
        rot_mat=cv2.getRotationMatrix2D((cenx,ceny),angle,1)

        img_rotated_by_alpha=cv2.warpAffine(I,rot_mat,(W,H))

        rot=rot_mat[:,0:2].T #shape (2,2)
        offset=rot_mat[:,2] #shape(2,)

        landmark_rotated=landmark.dot(rot)+offset
        return img_rotated_by_alpha,landmark_rotated