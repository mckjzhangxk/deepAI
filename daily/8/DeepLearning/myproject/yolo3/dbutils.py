import tensorflow as tf
import numpy as np
from iou import general_iou

def get_anchors(anchors_path, image_h, image_w):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = np.array(anchors.split(), dtype=np.float32)
    anchors = anchors.reshape(-1,2)
    anchors[:, 1] = anchors[:, 1] * image_h
    anchors[:, 0] = anchors[:, 0] * image_w
    return anchors.astype(np.int32)

def feature_map(y,
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

                z[r,c,k,2:4]=w,h
                # set logit=1,and class
                z[r, c, k,4]=1
                z[r, c, k,5+label] = 1
    return dtype(Z[0]),dtype(Z[1]),dtype(Z[2])
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
#
# z1,z2,z3=feature_map(y,13,416,
#               anchor_boxes=anchors_box,
#               num_classes=C,
#               )
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