import numpy  as np
import cv2

def get_label_boxes(I, y):
    '''
        y:(gy,gx,3,5+C),5:(cx,cy,cw,ch,label...),
            cx,cy表示相对与所分配单元格的偏移量,%
            cw,ch表示相对与所分配单元格的相对大小
        返回:
        x1,y1,x2,y2,anchor_idx,label:绝对的坐标,anchor_idx~[0-3),表示box属于那个box

    '''
    H, W = I.shape[0:2]
    gh, gw = y.shape[0:2]
    cellX, cellY = W // gw, H // gh

    logit = y[:, :, :, 4]  # (gh,gw,a)
    mask = logit > 0
    indices = np.where(mask)
    indices = np.array(indices)
    indices = indices.transpose()  # (#gt,3) 3=(i,j,a)

    # (#gt,5+C)
    labelobj = y[mask]

    y = cellY * indices[:, 0]
    x = cellX * indices[:, 1]

    labelobj_anchor = indices[:, 2]

    labelobj_cx = labelobj[:, 0] * cellX + x
    labelobj_cy = labelobj[:, 1] * cellY + y
    labelobj_w = labelobj[:, 2] * cellX
    labelobj_h = labelobj[:, 3] * cellY
    labelobj_classes = np.argmax(labelobj[:, 5:], axis=-1)

    labelobj_x1 = np.uint32(labelobj_cx - 0.5 * labelobj_w)
    labelobj_y1 = np.uint32(labelobj_cy - 0.5 * labelobj_h)
    labelobj_x2 = np.uint32(labelobj_cx + 0.5 * labelobj_w)
    labelobj_y2 = np.uint32(labelobj_cy + 0.5 * labelobj_h)

    return labelobj_x1, labelobj_y1, labelobj_x2, labelobj_y2, labelobj_anchor, labelobj_classes


def drawBox(I, x1, y1, x2, y2, anchor, classes, alpha=0.4):
    '''
    x1,y1,x2,y2绝对坐标,在这个图上画出区域大小,边框的颜色由anchor决定
    '''
    COLORS = [(128, 0, 0), (0, 128, 0), (0, 0, 128)]
    COLORS_CENTER = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    orgin = I

    n = len(x1)
    I = I.copy()
    for i in range(n):
        I = cv2.rectangle(I, (x1[i], y1[i]), (x2[i], y2[i]), COLORS[anchor[i]], -1)
        I = cv2.addWeighted(I, alpha, orgin, 1 - alpha, 0)
        I = cv2.circle(I, ((x1[i] + x2[i]) // 2, (y1[i] + y2[i]) // 2), 2, COLORS_CENTER[anchor[i]], 2, -1)
        I=cv2.putText(I,str(classes[i]),((x1[i] + x2[i]) // 2, (y1[i] + y2[i]) // 2),fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(0,0,0), thickness=2)
        orgin = I.copy()
    return I


def decodeGridImage(img, y, color, linewidth,classnames):
    '''
    y:(gh,gh,3,5+C),全都是相对与单元格的坐标
    把y转化成相对图片坐标后,在图上画出网格以及标注的box
    '''
    I = img.copy()
    H, W = I.shape[0:2]
    gH, gW = y.shape[0:2]
    cellx, celly = W // gW, H // gH
    #     horiziontal fix x,change y
    for r in range(1, gH):
        pt1 = (0, r * celly)
        pt2 = (W, r * celly)
        I = cv2.line(I, pt1, pt2, color, linewidth)
    # vertical fix y change x
    for c in range(1, gW):
        pt1 = (c * cellx, 0)
        pt2 = (c * cellx, H)
        I = cv2.line(I, pt1, pt2, color, linewidth)
    vx1, vy1, vx2, vy2, va, vclasses = get_label_boxes(I, y)
    vclasses=[classnames[c] for c in vclasses]
    I = drawBox(I, vx1, vy1, vx2, vy2, va, vclasses, alpha=0.6)
    return I


def decodeImage(image, y1, y2, y3, color=(255, 0, 0), linewidth=[5, 2, 1],classnames=None):
    '''
    
    :param image: 
    :param y1: 
    :param y2: 
    :param y3: 
    :param color: 网格的颜色
    :param linewidth: 网格的线宽
    :return: 
    '''
    image = np.uint8(image * 255)
    image13 = decodeGridImage(image, y1, color, linewidth[0],classnames)
    image26 = decodeGridImage(image, y2, color, linewidth[1],classnames)
    image52 = decodeGridImage(image, y3, color, linewidth[2],classnames)

    return image, image13, image26, image52


def anchorBox2Img(anchorbox,scale=0.2):
    COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    anchor_boxes = sorted(anchorbox, key=lambda x: x[0] * x[1])
    Ws = [int(box[0]*scale) for box in anchor_boxes]
    # 注意uint32,设置8溢出错误
    offset = np.cumsum(np.concatenate([[0.0], Ws])).astype(np.uint32)

    Hs = [int(box[1]*scale) for box in anchor_boxes]
    WW, HH = sum(Ws), max(Hs)
    I = np.zeros((HH, WW, 3), np.uint8)

    for i in range(len(Hs)):
        s = offset[i]
        pt1 = (s, 0)
        pt2 = (s + Ws[i], Hs[i])
        I = cv2.rectangle(I, pt1, pt2, COLORS[i % 3], -1)
        # I = cv2.putText(I, 'C%d' % i, (s + Ws[i] // 2 - 10, Hs[i] // 2), cv2.FONT_ITALIC, 2, 0.5, 3)
    # I=cv2.rotate(I,cv2.ROTATE_90_CLOCKWISE)
    return I