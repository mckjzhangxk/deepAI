from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import model.mtcnn_model as mtcnn
import numpy as np
import tensorflow as tf
import cv2


'''
    imap,reg 输入分别表示经过PNET得到的区域概率和修正
    imap:[H',W']
    reg:[H',W',4]
    scale表示原输入到PNET的图片缩放了 scale,所以还原会/scale
    t:>t的区域才会被保存

    :return (N',9)
    N'是筛选后的人脸数
    9:x1,y1,x2,y2,score,rx1,ry1,rx2,ry2
    这里的x1,x2,y1,y2是相对于原始图片的坐标!!
'''


def generateBoundingBox(imap, reg, scale, t):
    """Use heatmap to generate bounding boxes"""
    stride = 2
    cellsize = 12

    imap = np.transpose(imap)
    '''
    imap,dx1,dx2,dy1,dy2的第一个维度沿着W,第二个维度沿着H

    在imap的(i,j)映射回原图取区域后imap[i][j]表示这个区域是人脸的概率

    dx1[i][j]表示对左上角坐标沿着X(W)轴的修正!
    dy1[i][j]表示对左上角坐标沿着Y(H)轴的修正!

    dx2[i][j]表示对右下角坐标沿着X(W)轴的修正!
    dy2[i][j]表示对右下角坐标沿着Y(H)轴的修正!
    '''
    dx1 = np.transpose(reg[:, :, 0])
    dy1 = np.transpose(reg[:, :, 1])
    dx2 = np.transpose(reg[:, :, 2])
    dy2 = np.transpose(reg[:, :, 3])
    '''
    y(W)是第一个维度,x(H)是第二个维度
    '''
    y, x = np.where(imap >= t)
    if y.shape[0] == 1:
        dx1 = np.flipud(dx1)
        dy1 = np.flipud(dy1)
        dx2 = np.flipud(dx2)
        dy2 = np.flipud(dy2)
    score = imap[y, x]  # (N',)

    # vstack->(4,N') -->(N'4)
    reg = np.transpose(np.vstack([dx1[y, x], dy1[y, x], dx2[y, x], dy2[y, x]]))
    if reg.size == 0:
        reg = np.empty((0, 3))
    # bb是(N'2) bb[:,0]是X轴坐标,bb[:,1]是Y轴坐标,不要被变量命名干扰!
    bb = np.transpose(np.vstack([y, x]))
    q1 = np.fix((stride * bb + 1) / scale)
    q2 = np.fix((stride * bb + cellsize - 1 + 1) / scale)
    boundingbox = np.hstack([q1, q2, np.expand_dims(score, 1), reg])
    return boundingbox, reg

'''
boundingbox的修正,
boundingbox:(N,9)或者(N,4)
reg:(N,4)
return:(N,4)
'''
def bbreg(boundingbox,reg):
    """Calibrate bounding boxes"""
    if reg.shape[1]==1:
        reg = np.reshape(reg, (reg.shape[2], reg.shape[3]))

    w = boundingbox[:,2]-boundingbox[:,0]+1
    h = boundingbox[:,3]-boundingbox[:,1]+1
    b1 = boundingbox[:,0]+reg[:,0]*w
    b2 = boundingbox[:,1]+reg[:,1]*h
    b3 = boundingbox[:,2]+reg[:,2]*w
    b4 = boundingbox[:,3]+reg[:,3]*h
    boundingbox[:,0:4] = np.transpose(np.vstack([b1, b2, b3, b4 ]))
    return boundingbox

'''
对bboxA做了长宽修正,(w,h)->(l,l) l=max(w,h)
box中心没有变!,没有考虑是否超过边界
'''
def rerec(bboxA):
    """Convert bboxA to square."""
    h = bboxA[:,3]-bboxA[:,1]
    w = bboxA[:,2]-bboxA[:,0]
    l = np.maximum(w, h)
    bboxA[:,0] = bboxA[:,0]+w*0.5-l*0.5
    bboxA[:,1] = bboxA[:,1]+h*0.5-l*0.5
    bboxA[:,2:4] = bboxA[:,0:2] + np.transpose(np.tile(l,(2,1)))
    return bboxA


'''
    y,ey,x,ex:在原图片裁剪下的区域

    dy,edy,dx,edx:在目标(新图片)裁剪图片粘贴 到的区域
    tmpw,tmph新图片的大小

    对boxex进行修正,是他满足一下关系.
    tempw,temph分别表示没修正之前的box长宽.
    把这个box平移(-oldx+1,-oldy+1),得到新的矩形(1,1),(tempw,temph),这个标准矩形

    (x,y),(ex,ey)分别表示修正后左上角和右下角的坐标,x>=1,y>=1,ex<=w,ey<=h

    (dx,dy),(dex,dey),是上面矩形的标准化的变化,面积一样,
    例如x<0,x->1
    dx=1+(-x+1)=2-x
    x>w,x->w
    dex=tmpw+(-ex+w)
'''


def pad(total_boxes, w, h):
    """Compute the padding coordinates (pad the bounding boxes to square)"""
    #传入的坐标是[x1,x2),所以w=x2-x1不需要+1????
    tmpw = (total_boxes[:, 2] - total_boxes[:, 0]+1).astype(np.int32)
    tmph = (total_boxes[:, 3] - total_boxes[:, 1]+1).astype(np.int32)
    numbox = total_boxes.shape[0]

    dx = np.ones((numbox), dtype=np.int32)
    dy = np.ones((numbox), dtype=np.int32)
    edx = tmpw.copy().astype(np.int32)
    edy = tmph.copy().astype(np.int32)

    x = total_boxes[:, 0].copy().astype(np.int32)
    y = total_boxes[:, 1].copy().astype(np.int32)
    ex = total_boxes[:, 2].copy().astype(np.int32)
    ey = total_boxes[:, 3].copy().astype(np.int32)

    tmp = np.where(ex > w)
    #默认dx=1,dex=tmpw, 当ex的变化deltax=-ex+w,为了让尺寸一直de=tmpw+deltax
    edx.flat[tmp] = np.expand_dims(-ex[tmp] + w + tmpw[tmp], 1)
    ex[tmp] = w

    tmp = np.where(ey > h)
    edy.flat[tmp] = np.expand_dims(-ey[tmp] + h + tmph[tmp], 1)
    ey[tmp] = h
    # 默认dx=1,dex=tmpw, 当x的变化deltax=-x+1,为了让尺寸一直de=tmpw+deltax
    tmp = np.where(x < 1)
    dx.flat[tmp] = np.expand_dims(2 - x[tmp], 1)
    x[tmp] = 1

    tmp = np.where(y < 1)
    dy.flat[tmp] = np.expand_dims(2 - y[tmp], 1)
    y[tmp] = 1

    return dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph

def validbox(total_boxes):
    '''
    过滤长宽不合法的
    '''
    _w = total_boxes[:, 2] - total_boxes[:, 0]
    total_boxes = total_boxes[np.where(_w > 0)[0]]
    _h = total_boxes[:, 3] - total_boxes[:, 1]
    total_boxes = total_boxes[np.where(_h > 0)[0]]

    return total_boxes
def bb_landmark(box,landmarks):
    box=np.fix(np.expand_dims(box,axis=2))
    w=box[:,2]-box[:,0]+1
    h=box[:,3]-box[:,1]+1

    landmarks[:,0:9:2] = landmarks[:,0:9:2]  * w +box[:,0]
    landmarks[:,1:10:2]= landmarks[:,1:10:2] * h +box[:, 1]
    return landmarks
# function pick = nms(boxes,threshold,type)
def nms(boxes, threshold, method):
    if boxes.size==0:
        return np.empty((0,3))
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    s = boxes[:,4]
    area = (x2-x1+1) * (y2-y1+1)
    I = np.argsort(s)
    pick = np.zeros_like(s, dtype=np.int16)
    counter = 0
    while I.size>0:
        #每次选最后一个,因为分数最大
        i = I[-1]
        pick[counter] = i #记录
        counter += 1
        idx = I[0:-1] #其他的box 所有,排除i
        '''
        i是选择的box索引,idx是至今为止没有被选择的
        '''
        xx1 = np.maximum(x1[i], x1[idx])
        yy1 = np.maximum(y1[i], y1[idx])
        xx2 = np.minimum(x2[i], x2[idx])
        yy2 = np.minimum(y2[i], y2[idx])
        w = np.maximum(0.0, xx2-xx1+1)
        h = np.maximum(0.0, yy2-yy1+1)
        inter = w * h
        if method is 'Min':
            o = inter / np.minimum(area[i], area[idx])
        else:
            o = inter / (area[i] + area[idx] - inter)
        #np.where(o<=threshold)和选择区域相交<threshold进入下一轮
        I = I[np.where(o<=threshold)]
    pick = pick[0:counter]
    return pick





def imresample(img, sz):
    im_data = cv2.resize(img, (sz[1], sz[0]), interpolation=cv2.INTER_AREA) #@UndefinedVariable
    return im_data

    # This method is kept for debugging purpose

def normalize(im_data):
    return (im_data - 127.5) /128
'''
默认的PNET网络,输入是(N,W,H,C=3)
'''
def feedImage(image,transform=False):
    if image.ndim==3:
        image=np.expand_dims(image,axis=0)
    if transform:
        image=np.transpose(image,(0,2,1,3))
    return image


def pickfilter(rule, args):
    output=[]

    for x in args:
        if x is None:output.append(None)
        else:output.append(x[rule])
    return output

'''
orignImage:原始图片
total_boxes:(N',4)人脸在orignImage的坐标
size:输出图片尺寸
'''
def nextInput(orignImage, total_boxes, size):
    total_boxes=total_boxes.copy()
    h,w,_=orignImage.shape
    numBox = len(total_boxes)
    if numBox==0:return np.empty((0,size,size,3))

    '''
    make sure the index is integer,before pass it to pad function
    '''
    total_boxes[:, 0:4]=np.fix(total_boxes[:,0:4])

    '''
    y,ey,x,ex:在原图片裁剪下的区域
    
    dy,edy,dx,edx:在目标(新图片)裁剪图片粘贴 到的区域
    tmpw,tmph新图片的大小
    '''
    dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph=pad(total_boxes, w, h)


    target=np.empty((0,size,size,3),dtype=np.uint8)
    for n in range(numBox):
        _h,_w=int(tmph[n]),int(tmpw[n])
        # if _w<=0 or _h<=0:continue
        # print(_w,_h)
        tmp=np.zeros((_h,_w,3),dtype=np.uint8)
        tmp[dy[n] - 1:edy[n], dx[n] - 1:edx[n]]=orignImage[y[n]-1:ey[n],x[n]-1:ex[n]]
        tmp=imresample(tmp,(size,size))

        tmp=np.expand_dims(tmp,axis=0)

        target=np.append(target,tmp,axis=0)
    return target

def drawDectectBox(orignImage,total_boxes,scores=None):
    if isinstance(orignImage,str):
        orignImage=cv2.imread(orignImage)
        orignImage=cv2.cvtColor(orignImage,cv2.COLOR_BGR2RGB)

    h,w,_=orignImage.shape
    _tb = np.fix(total_boxes)
    dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(_tb, w, h)
    numBox = len(_tb)
    _I = orignImage.copy()
    for n in range(numBox):
        _I = cv2.rectangle(_I, (x[n] - 1, y[n] - 1), (ex[n] - 1, ey[n] - 1), (125, 199, 80), thickness=1)
        if scores is not None:
            sc=np.round(scores[n],3)
            pt=int(x[n]),int(y[n])
            cv2.putText(_I,str(sc),pt,cv2.FONT_HERSHEY_COMPLEX_SMALL,0.7,(255, 0, 0),1)
    return _I
def drawLandMarks(orignImage,landmarks):
    _I = orignImage.copy()
    numBox = len(landmarks)
    for n in range(numBox):
        for k in range(0,10,2):
            pt=int(landmarks[n][k]),int(landmarks[n][k+1])
            cv2.circle(_I,pt,5,(0,255,0),5)
    return _I


# if __name__ == '__main__':
#     sess = tf.Session()
#     pnet, rnet, onet = create_mtcnn(sess, '/home/zhangxk/AIProject/MTCNN_TRAIN/pnet/model/PNet-3')
#     image = cv2.imread('/home/zhangxk/AIProject/WIDER_train/images/0--Parade/0_Parade_Parade_0_904.jpg')
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     print('p_net---->totalbox and next input:')
#     totalbox,out=p_stage(image,minsize=50,pnet=pnet,factor=0.709,t=0.6,debug=True,nms_default=[0.5,0.7])
#     print(totalbox.shape)
#     print(out.shape)
#
#     image=drawDectectBox(image.copy(), totalbox, scores=None)
#     imsave('ssss.jpg',image)

# # saver=tf.train.Saver()
# # saver.restore(sess,'/home/zxk/PycharmProjects/deepAI/daily/8/studyOfFace/logs/models/facedect.ckpt-37')
# image=cv2.imread('../../images/zly.jpg')
# image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#
# print('p_net---->totalbox and next input:')
# totalbox,out=p_stage(image,minsize=50,pnet=pnet,factor=0.709,t=0.6,debug=True)
# print(totalbox.shape,out.shape)
#
# print('r_net---->totalbox and next input:')
# totalbox,out=r_stage(image,totalbox,out,rnet=rnet,t=0.6,debug=True)
# print(totalbox.shape,out.shape)
#
# print('o_net---->box,points,score:')
# totalbox,landmarks,score=o_stage(image,totalbox,out,onet=onet,t=0.7,debug=True)
# print(totalbox.shape,landmarks.shape,score.shape)
# print(score)