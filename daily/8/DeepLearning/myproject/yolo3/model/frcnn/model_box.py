# -*- coding: utf-8 -*-
# File: model_box.py
import numpy as np
import tensorflow as tf
from tensorpack.tfutils.scope_utils import under_name_scope


@under_name_scope()
def clip_boxes(boxes, window, name=None):
    """
    Args:
        boxes: nx4, xyxy
        window: [h, w]
    """
    boxes = tf.maximum(boxes, 0.0)
    m = tf.tile(tf.reverse(window, [0]), [2])    # (4,)
    boxes = tf.minimum(boxes, tf.cast(m, tf.float32), name=name)
    return boxes


@under_name_scope()
def decode_bbox_target(box_predictions, anchors,maxSize):
    """
    box_predictions:(tx,ty,tw,th),-->(x1,y1,x2,y2)
    anchors:(x1,y1,x2,y2),还原box_predictions为绝对坐标
    Args:
        box_predictions: (..., 4), logits,Resnet输出经过rpn后输出关于box的(h,w,a,4),还原称为绝对坐标
        anchors: (..., 4), floatbox. Must have the same shape,预定义的全局anchor box(h,w,a,4),绝对坐标

    Returns:
        box_decoded: (..., 4), float32. With the same shape.
    """
    orig_shape = tf.shape(anchors)
    box_pred_txtytwth = tf.reshape(box_predictions, (-1, 2, 2))
    box_pred_txty, box_pred_twth = tf.split(box_pred_txtytwth, 2, axis=1)
    # each is (...)x1x2
    anchors_x1y1x2y2 = tf.reshape(anchors, (-1, 2, 2))
    anchors_x1y1, anchors_x2y2 = tf.split(anchors_x1y1x2y2, 2, axis=1)

    waha = anchors_x2y2 - anchors_x1y1
    xaya = (anchors_x2y2 + anchors_x1y1) * 0.5

    clip = np.log(maxSize / 16.)
    wbhb = tf.exp(tf.minimum(box_pred_twth, clip)) * waha
    xbyb = box_pred_txty * waha + xaya
    x1y1 = xbyb - wbhb * 0.5
    x2y2 = xbyb + wbhb * 0.5    # (...)x1x2
    out = tf.concat([x1y1, x2y2], axis=-2)
    return tf.reshape(out, orig_shape)


@under_name_scope()
def encode_bbox_target(boxes, anchors):
    """
    Args:
        boxes: (..., 4), float32
        anchors: (..., 4), float32

    Returns:
        box_encoded: (..., 4), float32 with the same shape.
    
    坐标转化:
    boxes:(H,W,A,4),anchors:(H,W,A,4),boxes是我标注的在每一个可能"位置"的推荐区域大小位置
    anchors是在可能位置实际的anchor box大小位置,大小位置都是相对(0,0)的绝对坐标.
    boxes,anchor格式是x1,y1,x2,y2
    
    这个方法是把绝对转"相对",相对的是应该对于的anchor box,例如
    boxes(i,j,k)-->anchors(i,j,k)
    x1,y1,x2,y2,   ax1,ay1,ax2,ay2
    转化分2部分,绝对转相对,格式转化x1y1x2y2->txtytwth
        
    (x1,y1,x2,y2    )->(cx,cy,w,h    )
    (ax1,ay1,ax2,ay2)->(acx,acy,aw,ah)
    
    #长宽相对于anchor的长宽,中心相对于anchor的中心
    tw,th=log(w/aw),log(h/ah)
    tx,ty=(cx-ax)/aw,(cy-ay)/ah
    
    返回:(H,W,A,4) tx,ty,th,tw
    """
    anchors_x1y1x2y2 = tf.reshape(anchors, (-1, 2, 2)) #(HWA,2,2)axis=1表示坐标维度0->(x1,y1),1->(x2,y2)
    anchors_x1y1, anchors_x2y2 = tf.split(anchors_x1y1x2y2, 2, axis=1) #(HWA,1,2)
    waha = anchors_x2y2 - anchors_x1y1  #(HWA,2)
    xaya = (anchors_x2y2 + anchors_x1y1) * 0.5 #(HWA,2)

    boxes_x1y1x2y2 = tf.reshape(boxes, (-1, 2, 2)) #(HWA,2,2)
    boxes_x1y1, boxes_x2y2 = tf.split(boxes_x1y1x2y2, 2, axis=1) #(HWA,1,2)
    wbhb = boxes_x2y2 - boxes_x1y1
    xbyb = (boxes_x2y2 + boxes_x1y1) * 0.5

    # Note that here not all boxes are valid. Some may be zero
    txty = (xbyb - xaya) / waha
    twth = tf.log(wbhb / waha)  # may contain -inf for invalid boxes
    encoded = tf.concat([txty, twth], axis=1)  # (-1x2x2)
    return tf.reshape(encoded, tf.shape(boxes))


@under_name_scope()
def crop_and_resize(image, boxes, box_ind, crop_size, pad_border=True):
    """
    给出一批图片,一组boxes,在图片裁剪出boxes区域后,缩放成crop_size

    返回(n,C,cropsize,cropsize,),n是给出boxes的数量
    Aligned version of tf.image.crop_and_resize, following our definition of floating point boxes.

    Args:
        image: NCHW
        boxes: nx4, x1y1x2y2
        box_ind: (n,)
        crop_size (int):
    Returns:
        n,C,size,size
    """
    assert isinstance(crop_size, int), crop_size
    boxes = tf.stop_gradient(boxes)

    # TF's crop_and_resize produces zeros on border
    if pad_border:
        # this can be quite slow
        image = tf.pad(image, [[0, 0], [0, 0], [1, 1], [1, 1]], mode='SYMMETRIC')
        boxes = boxes + 1

    @under_name_scope()
    def transform_fpcoor_for_tf(boxes, image_shape, crop_shape):
        """
        (x1,y1,x2,y2)->(ny1,ny2,nx1,nx2)
        The way tf.image.crop_and_resize works (with normalized box):
        Initial point (the value of output[0]): x0_box * (W_img - 1)
        Spacing: w_box * (W_img - 1) / (W_crop - 1)
        Use the above grid to bilinear sample.

        However, what we want is (with fpcoor box):
        Spacing: w_box / W_crop
        Initial point: x0_box + spacing/2 - 0.5
        (-0.5 because bilinear sample (in my definition) assumes floating point coordinate
         (0.0, 0.0) is the same as pixel value (0, 0))

        This function transform fpcoor boxes to a format to be used by tf.image.crop_and_resize

        Returns:
            y1x1y2x2
        """
        x0, y0, x1, y1 = tf.split(boxes, 4, axis=1)

        spacing_w = (x1 - x0) / tf.cast(crop_shape[1], tf.float32)
        spacing_h = (y1 - y0) / tf.cast(crop_shape[0], tf.float32)

        imshape = [tf.cast(image_shape[0] - 1, tf.float32), tf.cast(image_shape[1] - 1, tf.float32)]
        nx0 = (x0 + spacing_w / 2 - 0.5) / imshape[1]
        ny0 = (y0 + spacing_h / 2 - 0.5) / imshape[0]

        nw = spacing_w * tf.cast(crop_shape[1] - 1, tf.float32) / imshape[1]
        nh = spacing_h * tf.cast(crop_shape[0] - 1, tf.float32) / imshape[0]

        return tf.concat([ny0, nx0, ny0 + nh, nx0 + nw], axis=1)

    # Expand bbox to a minium size of 1
    # boxes_x1y1, boxes_x2y2 = tf.split(boxes, 2, axis=1)
    # boxes_wh = boxes_x2y2 - boxes_x1y1
    # boxes_center = tf.reshape((boxes_x2y2 + boxes_x1y1) * 0.5, [-1, 2])
    # boxes_newwh = tf.maximum(boxes_wh, 1.)
    # boxes_x1y1new = boxes_center - boxes_newwh * 0.5
    # boxes_x2y2new = boxes_center + boxes_newwh * 0.5
    # boxes = tf.concat([boxes_x1y1new, boxes_x2y2new], axis=1)

    image_shape = tf.shape(image)[2:]
    boxes = transform_fpcoor_for_tf(boxes, image_shape, [crop_size, crop_size])
    image = tf.transpose(image, [0, 2, 3, 1])  # nhwc
    ret = tf.image.crop_and_resize(
        image, boxes, tf.cast(box_ind, tf.int32),
        crop_size=[crop_size, crop_size])
    ret = tf.transpose(ret, [0, 3, 1, 2])  # ncss
    return ret

@under_name_scope()
def pairwise_iou(boxlist1, boxlist2):
    """Computes pairwise intersection-over-union between box collections.

    Args:
      boxlist1: Nx4 floatbox
      boxlist2: Mx4

    Returns:
      a tensor with shape [N, M] representing pairwise iou scores.
    """

    @under_name_scope()
    def area(boxes):
        """
        Args:
          boxes: nx4 floatbox

        Returns:
          n
        """
        x_min, y_min, x_max, y_max = tf.split(boxes, 4, axis=1)
        return tf.squeeze((y_max - y_min) * (x_max - x_min), [1])

    @under_name_scope()
    def pairwise_intersection(boxlist1, boxlist2):
        """Compute pairwise intersection areas between boxes.

        Args:
          boxlist1: Nx4 floatbox
          boxlist2: Mx4

        Returns:
          a tensor with shape [N, M] representing pairwise intersections
        """
        x_min1, y_min1, x_max1, y_max1 = tf.split(boxlist1, 4, axis=1)
        x_min2, y_min2, x_max2, y_max2 = tf.split(boxlist2, 4, axis=1)
        all_pairs_min_ymax = tf.minimum(y_max1, tf.transpose(y_max2))
        all_pairs_max_ymin = tf.maximum(y_min1, tf.transpose(y_min2))
        intersect_heights = tf.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
        all_pairs_min_xmax = tf.minimum(x_max1, tf.transpose(x_max2))
        all_pairs_max_xmin = tf.maximum(x_min1, tf.transpose(x_min2))
        intersect_widths = tf.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
        return intersect_heights * intersect_widths

    intersections = pairwise_intersection(boxlist1, boxlist2)
    areas1 = area(boxlist1)
    areas2 = area(boxlist2)
    unions = (
        tf.expand_dims(areas1, 1) + tf.expand_dims(areas2, 0) - intersections)
    return tf.where(
        tf.equal(intersections, 0.0),
        tf.zeros_like(intersections), tf.truediv(intersections, unions))

@under_name_scope()
def roi_align(featuremap, boxes, resolution):
    """
    Args:
        featuremap: 1xCxHxW
        boxes: Nx4 floatbox
        resolution: output spatial resolution
    根据boxes给出区域,在相应featuremap 裁剪出后,在缩放称为resolution.
    实现细节:
        先采样boxex,然后缩放成2*resolution.最后在avg成resolution
    Returns:
        NxCx res x res
    """
    # sample 4 locations per roi bin
    ret = crop_and_resize(
        featuremap, boxes,
        tf.zeros([tf.shape(boxes)[0]], dtype=tf.int32),
        resolution * 2)
    ret = tf.nn.avg_pool(ret, [1, 1, 2, 2], [1, 1, 2, 2], padding='SAME', data_format='NCHW')
    return ret