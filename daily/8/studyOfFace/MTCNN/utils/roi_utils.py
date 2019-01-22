import numpy as np
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

facebox:人脸的绝对坐标(4,)x1,y1,x2,y2
landmark:五官坐标,要有10个元素
返回:[5,2]的np array
'''
def GetLandMarkPoint(facebox,landmark):
    if isinstance(landmark,list):
        landmark=np.array(landmark)
    landmark=landmark.copy()
    if landmark.shape!=(5,2):
        landmark=np.reshape(landmark,(5,2))
    w=facebox[2]-facebox[0]
    h=facebox[3]-facebox[1]
    x1,y1=facebox[0],facebox[1]

    landmark[:,0]=(landmark[:,0]-x1)/w
    landmark[:,1]=(landmark[:,1]-y1)/h
    return landmark