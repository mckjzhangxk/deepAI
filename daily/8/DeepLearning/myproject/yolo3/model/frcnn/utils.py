import numpy as np
from model.frcnn.config import config as cfg
from tensorpack.utils.argtools import memoized

def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2 ** np.arange(3, 6)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.

    生成程序需要的anchorbox,基本的anchor w=h=basesize,生成不同scale,不同ratio的
    anchor box返回 (12,4)个anchor box
    """

    base_anchor = np.array([1, 1, base_size, base_size], dtype='float32') - 1
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in range(ratio_anchors.shape[0])])
    return anchors


def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    输入的anchor是(x1,y1,x2,y2)格式
    返回的是(w,h,cx,cy)格式
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    (cx,cy,w,h)-->(x1,y1,x2,y1)
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors


def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    anchor是参照的anchor_box,给出一组ratios{h/w},返回一组W/H 是ratios,面积
    是anchor的 anchors
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    anchor的中心点不变,长宽缩放scales
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors
@memoized
def get_all_anchors(stride=None, sizes=None):
    """
    Get all anchors in the largest possible image, shifted, floatbox
    Args:
        stride (int): the stride of anchors.
        sizes (tuple[int]): the sizes (sqrt area) of anchors

    Returns:
        anchors: SxSxNUM_ANCHORx4, where S == ceil(MAX_SIZE/STRIDE), floatbox
        The layout in the NUM_ANCHOR dim is NUM_RATIO x NUM_SIZE.
        stride:是Resnet50 作用后尺寸缩放比,对于  ResC4,stride=16,size表示基准anchor的边长,
        这里设在的是(32,64,128,256),ratio一般是(.5,1,2),对于某一个位置, anchor box的数量
        是NUM_RATIO x NUM_SIZE.MAX_SIZE=1333是允许图片最大的边长,最大图片那么经过NN后,就会有
        fieldsizexfieldsize的feature map,fieldsize=ceil(MAX_SIZE/STRIDE)

        返回(fieldsize,fieldsize,NUM_ANCHOR,4)的numpy,表示所有可能位置的所有可能anchor box
        的绝对坐标,(x1,y1,x2,y2)格式?

    """
    if stride is None:
        stride = cfg.RPN.ANCHOR_STRIDE
    if sizes is None:
        sizes = cfg.RPN.ANCHOR_SIZES
    # Generates a NAx4 matrix of anchor boxes in (x1, y1, x2, y2) format. Anchors
    # are centered on stride / 2, have (approximate) sqrt areas of the specified
    # sizes, and aspect ratios as given.
    cell_anchors = generate_anchors(
        stride,
        scales=np.array(sizes, dtype=np.float) / stride,
        ratios=np.array(cfg.RPN.ANCHOR_RATIOS, dtype=np.float))
    # anchors are intbox here.
    # anchors at featuremap [0,0] are centered at fpcoor (8,8) (half of stride)

    max_size = cfg.PREPROC.MAX_SIZE
    field_size = int(np.ceil(max_size / stride))
    shifts = np.arange(0, field_size) * stride
    shift_x, shift_y = np.meshgrid(shifts, shifts)
    shift_x = shift_x.flatten()
    shift_y = shift_y.flatten()
    shifts = np.vstack((shift_x, shift_y, shift_x, shift_y)).transpose()
    # Kx4, K = field_size * field_size
    K = shifts.shape[0]

    A = cell_anchors.shape[0]
    field_of_anchors = (
        cell_anchors.reshape((1, A, 4)) +
        shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    field_of_anchors = field_of_anchors.reshape((field_size, field_size, A, 4))
    # FSxFSxAx4
    # Many rounding happens inside the anchor code anyway
    # assert np.all(field_of_anchors == field_of_anchors.astype('int32'))
    field_of_anchors = field_of_anchors.astype('float32')
    field_of_anchors[:, :, :, [2, 3]] += 1
    return field_of_anchors