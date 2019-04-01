import tensorflow as tf
import numpy as np
from iou import general_iou
'''
1.修改feature_map,使他更高效
2.修改pyfunc->pyfunction

'''
def load_anchor_boxes(anchors_path, image_h, image_w):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = np.array(anchors.split(), dtype=np.float32)
    anchors = anchors.reshape(-1,2)
    anchors[:, 1] = anchors[:, 1] * image_h
    anchors[:, 0] = anchors[:, 0] * image_w
    return anchors.astype(np.int32)
def feature_map_v2(y,
                   grid_size,
                   image_size,
                   anchor_boxes,
                   num_classes,
                   dtype=np.float32):
    def filter_mask(X,mask):
        return [x[mask] for x in X]

    num_anchors = len(anchor_boxes) // 3
    Z=[np.zeros(((2**l)*grid_size, (2**l)*grid_size, num_anchors, 5 + num_classes))
       for l in range(3)]

    cx,cy=(y[:,0]+y[:,2])/2,(y[:,1]+y[:,3])/2
    w,h=y[:,2]-y[:,0],y[:,3]-y[:,1]
    label=y[:,4].astype(np.int32)
    validmask=(w>0) * (h>0)

    cx,cy,w,h,label=filter_mask([cx,cy,w,h,label],validmask)


    boxsize=np.stack((w,h),axis=1)
    iou=general_iou(boxsize,anchor_boxes)
    bestAnchor = np.argmax(iou,axis=-1)

    for i,z in enumerate(Z):
        mapsize=grid_size*(2**i)
        cellsize=image_size//mapsize

        x=cx/cellsize
        y=cy/cellsize

        r=np.floor(y).astype(np.int32)
        c=np.floor(x).astype(np.int32)
        ancharmask=(bestAnchor>=6-3*i) * (bestAnchor<9-3*i)
        x,y,r,c,k,_w,_h,_label=filter_mask([x,y,r,c,bestAnchor,w,h,label],ancharmask)
        if len(x)==0:continue
        k=k%3

        z[r,c,k,0]=x-c
        z[r,c,k,1]=y-r
        z[r,c,k,2]=_w/cellsize
        z[r,c,k,3]=_h/cellsize
        z[r,c,k,4]=1
        z[r,c,k,_label+5]=1
    return dtype(Z[0]), dtype(Z[1]), dtype(Z[2])
def feature_map_v1(y,
                   grid_size,
                   image_size,
                   anchor_boxes,
                   num_classes,
                   dtype=np.float32):
    '''
    y:输入的标注对象信息,有gtbox个,对于每个box信息,计算box的大小,与全局的anchor_box比较,
    确定标注的box最接近那个anchorbox,然后在那个anchorbox对应的热力图标注出来,对于关系
    [6,7,8]--->(grid_size,grid_size)
    [3,4,5]--->(2*grid_size,2*grid_size)
    [0,1,2]--->(4*grid_size,4*grid_size)
    
    :param y: (#gtbox,5) 2+2+1 (x1,y1,x2,y2,label)
    :param grid_size: 13 or 19
    :param image_size: 416 or 608
    :param num_anchors: 3
    :param anchor_boxes: all anchors (3*num_anchors,2)
    :param num_classes: 
    :return: 
    '''
    def _get_anchor_mask():
        anchorindex=np.arange(len(anchor_boxes))
        anchorindex =anchorindex.reshape((3,-1))
        anchorindex=anchorindex[::-1]
        return anchorindex.tolist()
    ANCHOR_MASK=_get_anchor_mask()

    num_anchors=len(anchor_boxes)//3
    Z=[np.zeros(((2**l)*grid_size, (2**l)*grid_size, num_anchors, 5 + num_classes))
       for l in range(3)]
    CELL_SIZES=[image_size// ((2**l)*grid_size) for l in range(3)]

    for i in range(0,len(y)):
        x1,y1,x2,y2,label=y[i]
        label=int(label)

        cx,cy=(x1+x2)/2,(y1+y2)/2
        w, h = x2 - x1, y2 - y1
        if w<0 or h<0 :continue
        iou=general_iou(anchor_boxes,np.array([w,h]))
        bestAnchor=np.argmax(iou)
        for l in range(3):
            z=Z[l]
            if bestAnchor in ANCHOR_MASK[l]:
                k=ANCHOR_MASK[l].index(bestAnchor)

                r,c=int(np.floor(cy/CELL_SIZES[l])),int(np.floor(cx/CELL_SIZES[l]))
                cy,cx=cy/CELL_SIZES[l]-r,cx/CELL_SIZES[l]-c

                # set center of box
                z[r,c,k,0:2]=cx,cy
                # set ralative size to chioced anchor box

                z[r,c,k,2:4]=w/CELL_SIZES[l],h/CELL_SIZES[l]
                # set logit=1,and class
                z[r, c, k,4]=1
                z[r, c, k,5+label] = 1
    return dtype(Z[0]),dtype(Z[1]),dtype(Z[2])

class ImageDataset():
    def __init__(self,gridsize=13,imagesize=416,anchor_boxes=None,num_classes=80):
        self.gridsize=gridsize
        self.imagesize=imagesize
        self.anchor_boxes=anchor_boxes
        self.num_classes=num_classes

    def build_example(self, filepath, batch_size=1, epoch=1, shuffle=True, parallels=4,eager=False):
        '''
        
        :param filepath: 
        :param batch_size: 
        :param epoch: 
        :return: 
        '''
        dataset=tf.data.TextLineDataset(filepath)
        if shuffle:
            dataset=dataset.shuffle(4*batch_size)

        dataset=dataset.map(self._retrive,parallels)
        dataset = dataset.map(self._resizeImage, parallels)
        dataset = dataset.map(self._featureMap, parallels)

        dataset=dataset.repeat(epoch)
        dataset=dataset.batch(batch_size)

        iterator=dataset.make_one_shot_iterator()
        if eager:return iterator

        image,y13,y26,y52=iterator.get_next()
        return image,y13,y26,y52
        # return iterator

    def _retrive(self,line):
        sps=tf.string_split([line]).values
        path=sps[0]
        content=tf.read_file(path)
        image=tf.image.decode_jpeg(content,3)
        image=tf.to_float(image)/255.0

        boxes=tf.string_to_number(sps[1:])
        boxes=tf.reshape(boxes,(-1,5))

        return image,boxes

    def _resizeImage(self,image,boxes):
        imshape=tf.to_float(tf.shape(image))
        h,w=imshape[0],imshape[1]

        image=tf.image.resize_images(image,(self.imagesize,self.imagesize))

        x1=boxes[:,0]*self.imagesize/w
        x2=boxes[:,2]*self.imagesize/w

        y1=boxes[:,1]*self.imagesize/h
        y2=boxes[:,3]*self.imagesize/h
        label=boxes[:,4]
        boxes=tf.stack([x1,y1,x2,y2,label],axis=1)
        return image,boxes

    def _featureMap(self,image,boxes):
        y1,y2,y3=tf.py_func(feature_map_v2,
                            [boxes,self.gridsize,self.imagesize,self.anchor_boxes,self.num_classes],
                            [tf.float32,tf.float32,tf.float32])
        return image,y1,y2,y3
#
# filepath='/home/zhangxk/projects/deepAI/daily/8/DeepLearning/myproject/yolo3/data/sample.txt'
# anchorpath='/home/zhangxk/projects/deepAI/daily/8/DeepLearning/myproject/yolo3/data/raccoon_my_anchors.txt'
# anchorbox=load_anchor_boxes(anchorpath,416,416)
#
# db=ImageDataset(anchor_boxes=anchorbox.tolist())

# tf.enable_eager_execution()
# tf.executing_eagerly()
# iterator=db.build_example(filepath,batch_size=1,epoch=2,shuffle=False)
# KK=1000
# image,y1,y2,y3=db.build_example(filepath,batch_size=5,epoch=KK,shuffle=False)
# with tf.Session() as sess:
#     import time
#     st=time.time()
#     for i in range(KK):
#         # images,y1,y2,y3=iterator.next()
#         # print(images.shape)
#         # print(y1.shape,y2.shape,y3.shape)
#         _image, _y1, _y2, _y3=sess.run([image,y1,y2,y3])
#         # print(_image.shape)
#         # print(_y1.shape,_y2.shape,_y3.shape)
#     ed=time.time()
#     print((ed-st)/KK)
    # (1, 203, 248, 3)
    # tf.Tensor([[[6.   7. 240. 157.   0.]]], shape=(1, 1, 5), dtype=float32)
    # (1, 253, 199, 3)
    # tf.Tensor([[[27.  11. 194. 228.   0.]]], shape=(1, 1, 5), dtype=float32)

from drawutils import decodeImage
class ImageBrower():
    def __init__(self,annpath,anchorboxpath,C=1):
        anchor_boxes = load_anchor_boxes(anchorboxpath, 416, 416)
        db = ImageDataset(gridsize=13, imagesize=416, anchor_boxes=anchor_boxes, num_classes=C)
        self.iterator = db.build_example(annpath,
                                    batch_size=32,
                                    epoch=1000,
                                    shuffle=False,
                                    parallels=4,
                                    eager=True)
        self.cursor=-1
    def next(self):
        if self.cursor==-1 or self.cursor>len(self.img):
            img, y1, y2, y3 = self.iterator.next()
            self.img = img.numpy()
            self.y1 = y1.numpy()
            self.y2 = y2.numpy()
            self.y3 = y3.numpy()
            self.cursor=-1
        self.cursor+=1

        ii=self.img[self.cursor]
        i1=self.y1[self.cursor]
        i2=self.y2[self.cursor]
        i3=self.y3[self.cursor]

        return decodeImage(ii,i1,i2,i3,(255, 255, 0))
    def prev(self):
        if self.img is not None:
            self.cursor-=1
            ii=self.img[self.cursor]
            i1=self.y1[self.cursor]
            i2=self.y2[self.cursor]
            i3=self.y3[self.cursor]
            return decodeImage(ii,i1,i2,i3,(255, 255, 0))
# N,C=22,6
# xy=np.random.rand(N,2)
# xy1=xy+np.random.rand(N,2)*20
# label=np.random.randint(0,C,size=(N,1))
# y=np.concatenate([xy,xy1,label],axis=-1)
#
#
# anchors_box=np.random.rand(9,2)
# anchors_box=sorted(anchors_box,key=lambda x:x[0]*x[1])
# anchors_box=np.array(anchors_box)
# import time
#
# st=time.time()
# KK=1000
# for i in range(KK):
#     z1,z2,z3=feature_map_v1(y, 13, 416,
#                             anchor_boxes=anchors_box,
#                             num_classes=C,
#                             )
# ed=time.time()
# print((ed-st)/KK)
#
# c=np.sum(z1[...,4])+np.sum(z2[...,4])+np.sum(z3[...,4])
# print(c)

# anchor_boxes=np.random.rand(3*6,2)
# def _get_anchor_mask():
#     anchorindex=np.arange(len(anchor_boxes))
#     anchorindex =anchorindex.reshape((3,-1))
#     anchorindex=anchorindex[::-1]
#     return anchorindex.tolist()
# print(_get_anchor_mask())


