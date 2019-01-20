from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import model.mtcnn_model as mtcnn
import numpy as np
import tensorflow as tf
import cv2
from scipy.misc import imsave

'''
这应该是本模块对外提供的唯一方法:
    输入:filename:
        minsize:最小尺寸
        factor:金字塔缩放参数
        thresholds:经过3个网络 放行概率
        
    返回:np.array([N',5]),
        9表示x1,y1,x2,y2,score

'''
def detect_face(filename,minsize,factor,thresholds=[0.3,0.1,0.4],sess=None,models=[None,None,None]):
    pnet_path, rnet_path, onet_path=models
    pnet,rnet,onet=create_mtcnn(sess,pnet_path,rnet_path,onet_path)

    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    assert image is not None,'%s not exist'%filename


    if pnet:
        total_box,_=p_stage(image,minsize,pnet,factor,thresholds[0],nms_default,False)
        if len(total_box)>0:
            total_box=total_box[:,0:4] #返回的是9维的,只要前4

def create_mtcnn(sess,pnet_path=None,rnet_path=None,onet_path=None):
    pnet_fun, rnet_fun, onet_fun=None,None,None

    if pnet_path:
        # with tf.variable_scope('pnet'):
        data = tf.placeholder(tf.float32, (None,None,None,3))
        prob_tensor, regbox_tensor=mtcnn.createPNet(data, False)
        saver = tf.train.Saver()
        saver.restore(sess, pnet_path)
        pnet_fun = lambda img: sess.run((regbox_tensor,prob_tensor), feed_dict={data: img})

    if rnet_path:
        with tf.variable_scope('rnet'):
            data = tf.placeholder(tf.float32, (None,24,24,3), 'input')
        mtcnn.createPNet(data, False)
    if onet_path:
        with tf.variable_scope('onet'):
            data = tf.placeholder(tf.float32, (None,48,48,3), 'input')
        mtcnn.createPNet(data, False)

    # pnet_fun = lambda img : sess.run(('pnet/conv4-2/BiasAdd:0', 'pnet/prob1:0'), feed_dict={'pnet/input:0':img})
    # rnet_fun = lambda img : sess.run(('rnet/conv5-2/conv5-2:0', 'rnet/prob1:0'), feed_dict={'rnet/input:0':img})
    # onet_fun = lambda img : sess.run(('onet/conv6-2/conv6-2:0', 'onet/conv6-3/conv6-3:0', 'onet/prob1:0'), feed_dict={'onet/input:0':img})
    return pnet_fun, rnet_fun, onet_fun


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

    (x,y),(ex,ey)分别表示修正后左上角和右下角的坐标,x>=1,y>=1,ex<=w,ey<=y

    (dx,dy),(dex,dey),是上面矩形的标准化的变化,面积一样,
    例如x<0,x->1
    dx=1+(-x+1)=2-x
    x>w,x->w
    dex=tmpw+(-ex+w)
'''


def pad(total_boxes, w, h):
    """Compute the padding coordinates (pad the bounding boxes to square)"""
    tmpw = (total_boxes[:, 2] - total_boxes[:, 0] + 1).astype(np.int32)
    tmph = (total_boxes[:, 3] - total_boxes[:, 1] + 1).astype(np.int32)
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
    edx.flat[tmp] = np.expand_dims(-ex[tmp] + w + tmpw[tmp], 1)
    ex[tmp] = w

    tmp = np.where(ey > h)
    edy.flat[tmp] = np.expand_dims(-ey[tmp] + h + tmph[tmp], 1)
    ey[tmp] = h

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

    landmarks[:,0:5]=landmarks[:,0:5]*w+box[:,0]
    landmarks[:, 5:10] = landmarks[:, 5:10] * h + box[:, 1]

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


def filter(rule,args):
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
def outputImage2NextStage(orignImage,total_boxes,size):
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
        for k in range(5):
            pt=landmarks[n][k],landmarks[n][k+5]
            cv2.circle(_I,pt,1,(0,255,0),2)
    return _I

'''
这里有2次调用NMS,参数分别是0.5,0.6
分别是对一张图的所有box的阈值和对 所有scale图片处理完成,保存的box的阈值
return:total_boxes (N'4)
       pout  (N',24,24,3)
'''
def p_stage(img,minsize,pnet,factor,t,debug=False,nms_default=[0.5,0.6],cut_image=True):

    h,w,_=img.shape
    l=min(h,w)

    '''
    minsize is min size of a face,so scale origin image to by factor 12/minsize,
    then pass it to a pnet,as least get a output(1x1x4,1x1,2),because pnet is a
    detector with window size =12
    '''
    scale=12/minsize
    scales=[]
    while int(l*scale)>=12:
        scales.append(scale)
        scale*=factor

    total_boxes = np.empty((0, 9))
    pout = np.empty((0, 24, 24, 3))
    #不同的scale经过筛选后的total_box没有经过修正
    for i,scale in enumerate(scales):
        hs,ws=int(np.ceil(scale*h)),int(np.ceil(scale*w))

        im_data=imresample(img,(hs,ws))
        if debug:
            imsave('debug/my%d.jpg'%i,im_data)

        im_data=normalize(im_data)
        img_x = feedImage(im_data)
        out=pnet(img_x)
        #out[0]是regressor,0表示第一张图片,shape[H',W',4]
        regressor=out[0][0]
        #out[1]是prosibility,out[1][0]是第一张图片,shape[H',W',2]
        score = out[1][0][:,:,1]
        #bbox是原图的坐标,(N',9),N'是概率>t保留下来的人脸数量
        bbox,_=generateBoundingBox(score.copy(),regressor.copy(),scale,t)
        if bbox.size>1:
            pick=nms(bbox.copy(),nms_default[0],'Union')
            bbox=bbox[pick]
            total_boxes = np.append(total_boxes, bbox, axis=0)

    if len(total_boxes)>1:
        pick=nms(total_boxes.copy(),nms_default[1],'Union')
        total_boxes = total_boxes[pick]
        '''
        using regress to adjust
        '''
        total_boxes=bbreg(total_boxes,total_boxes[:,5:9])
        total_boxes = rerec(total_boxes.copy())
        '''
        进一步过滤长宽不合法的
        '''
        _w = total_boxes[:,2]- total_boxes[:,0]
        total_boxes=total_boxes[np.where(_w>0)[0]]

        if not cut_image: return (total_boxes, None)
        pout=outputImage2NextStage(img,total_boxes,24)



        '''
            show target image after pnet
        '''
        if debug:
            _I=drawDectectBox(img,total_boxes)
            imsave('debug/pnet.jpg',_I)

    return total_boxes,pout


def r_stage(orignimage, total_boxes, images, rnet, t, debug=False):
    if len(total_boxes)==0:
        return np.empty((0,5)),np.empty((0,10))

    tempings=normalize(images)
    tempings=feedImage(tempings)
    out=rnet(tempings)

    #regerssor:(#,4),score:(#,1)
    regressor,score=out[0],out[1][:,1]

    ipass=score>=t
    total_boxes,regressor,score=filter(ipass,[total_boxes,regressor,score])

    if len(total_boxes)>0:
        pick=nms(total_boxes,0.7,'Union')
        total_boxes, regressor, score = filter(pick,[total_boxes, regressor, score])

        total_boxes[:,0:4] = rerec(bbreg(total_boxes[:,0:4], regressor))
        total_boxes[:, 4] = score
        total_boxes[:,5:9]=regressor

    rout=outputImage2NextStage(orignimage,total_boxes,48)
    if debug:
        if debug:
            _I=drawDectectBox(orignimage,total_boxes)
            imsave('debug/rnet.jpg',_I)

    return total_boxes,rout


def o_stage(orignimage, total_boxes, images, onet, t, debug=False):
    if len(total_boxes)==0:
        return np.empty((0,4)),np.empty((0,10)),np.empty((0))

    tempings=normalize(images)
    tempings=feedImage(tempings)
    out=onet(tempings)

    regressor,landmarks,score=out[0],out[1],out[2][:,1]

    ipass=score>=t
    total_boxes, regressor, landmarks, score=filter(ipass,[total_boxes,regressor,landmarks,score])

    boxes=np.empty((0,4))
    if len(total_boxes)>0:
        h,w,_=orignimage.shape
        total_boxes=bbreg(total_boxes.copy(),regressor)

        pick=nms(total_boxes,0.7,'Min')
        total_boxes, regressor, landmarks, score = filter(pick, [total_boxes, regressor, landmarks, score])
        dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph=pad(total_boxes,w,h)
        boxes=np.stack([x,y,ex,ey],axis=1)

        landmarks=bb_landmark(boxes,landmarks)
        if debug:
            _I=drawDectectBox(orignimage,boxes)
            _I=drawLandMarks(_I,landmarks)
            imsave('debug/onet.jpg',_I)
    return boxes,landmarks,score



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