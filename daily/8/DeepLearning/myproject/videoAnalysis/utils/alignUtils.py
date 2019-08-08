import numpy as np
import cv2
from PIL import Image
# reference facial points, a list of coordinates (x,y)
REFERENCE_FACIAL_POINTS = [
    [30.29459953,  51.69630051],
    [65.53179932,  51.50139999],
    [48.02519989,  71.73660278],
    [33.54930115,  92.3655014],
    [62.72990036,  92.20410156]
]

DEFAULT_CROP_SIZE = (96, 112)



class FaceWarpException(Exception):
    def __str__(self):
        return 'In File {}:{}'.format(
            __file__, super.__str__(self))

def get_reference_facial_points(output_size=None,
                                inner_padding_factor=0.0,
                                outer_padding=(0, 0),
                                default_square=False):
    """
    Function:
    ----------
        get reference 5 key points according to crop settings:
        0. Set default crop_size:
            if default_square: 
                crop_size = (112, 112)
            else: 
                crop_size = (96, 112)
        1. Pad the crop_size by inner_padding_factor in each side;
        2. Resize crop_size into (output_size - outer_padding*2),
            pad into output_size with outer_padding;
        3. Output reference_5point;
    Parameters:
    ----------
        @output_size: (w, h) or None
            size of aligned face image
        @inner_padding_factor: (w_factor, h_factor)
            padding factor for inner (w, h)
        @outer_padding: (w_pad, h_pad)
            each row is a pair of coordinates (x, y)
        @default_square: True or False
            if True:
                default crop_size = (112, 112)
            else:
                default crop_size = (96, 112);
        !!! make sure, if output_size is not None:
                (output_size - outer_padding) 
                = some_scale * (default crop_size * (1.0 + inner_padding_factor))
    Returns:
    ----------
        @reference_5point: 5x2 np.array
            each row is a pair of transformed coordinates (x, y)
    """
    #print('\n===> get_reference_facial_points():')

    #print('---> Params:')
    #print('            output_size: ', output_size)
    #print('            inner_padding_factor: ', inner_padding_factor)
    #print('            outer_padding:', outer_padding)
    #print('            default_square: ', default_square)

    tmp_5pts = np.array(REFERENCE_FACIAL_POINTS)
    tmp_crop_size = np.array(DEFAULT_CROP_SIZE)

    # 0) make the inner region a square
    if default_square:
        size_diff = max(tmp_crop_size) - tmp_crop_size
        tmp_5pts += size_diff / 2
        tmp_crop_size += size_diff

    #print('---> default:')
    #print('              crop_size = ', tmp_crop_size)
    #print('              reference_5pts = ', tmp_5pts)

    if (output_size and
            output_size[0] == tmp_crop_size[0] and
            output_size[1] == tmp_crop_size[1]):
        #print('output_size == DEFAULT_CROP_SIZE {}: return default reference points'.format(tmp_crop_size))
        return tmp_5pts

    if (inner_padding_factor == 0 and
            outer_padding == (0, 0)):
        if output_size is None:
            #print('No paddings to do: return default reference points')
            return tmp_5pts
        else:
            raise FaceWarpException(
                'No paddings to do, output_size must be None or {}'.format(tmp_crop_size))

    # check output size
    if not (0 <= inner_padding_factor <= 1.0):
        raise FaceWarpException('Not (0 <= inner_padding_factor <= 1.0)')

    if ((inner_padding_factor > 0 or outer_padding[0] > 0 or outer_padding[1] > 0)
            and output_size is None):
        output_size = tmp_crop_size * \
            (1 + inner_padding_factor * 2).astype(np.int32)
        output_size += np.array(outer_padding)
        #print('              deduced from paddings, output_size = ', output_size)

    if not (outer_padding[0] < output_size[0]
            and outer_padding[1] < output_size[1]):
        raise FaceWarpException('Not (outer_padding[0] < output_size[0]'
                                'and outer_padding[1] < output_size[1])')

    # 1) pad the inner region according inner_padding_factor
    #print('---> STEP1: pad the inner region according inner_padding_factor')
    if inner_padding_factor > 0:
        size_diff = tmp_crop_size * inner_padding_factor * 2
        tmp_5pts += size_diff / 2
        tmp_crop_size += np.round(size_diff).astype(np.int32)

    #print('              crop_size = ', tmp_crop_size)
    #print('              reference_5pts = ', tmp_5pts)

    # 2) resize the padded inner region
    #print('---> STEP2: resize the padded inner region')
    size_bf_outer_pad = np.array(output_size) - np.array(outer_padding) * 2
    #print('              crop_size = ', tmp_crop_size)
    #print('              size_bf_outer_pad = ', size_bf_outer_pad)

    if size_bf_outer_pad[0] * tmp_crop_size[1] != size_bf_outer_pad[1] * tmp_crop_size[0]:
        raise FaceWarpException('Must have (output_size - outer_padding)'
                                '= some_scale * (crop_size * (1.0 + inner_padding_factor)')

    scale_factor = size_bf_outer_pad[0].astype(np.float32) / tmp_crop_size[0]
    #print('              resize scale_factor = ', scale_factor)
    tmp_5pts = tmp_5pts * scale_factor
#    size_diff = tmp_crop_size * (scale_factor - min(scale_factor))
#    tmp_5pts = tmp_5pts + size_diff / 2
    tmp_crop_size = size_bf_outer_pad
    #print('              crop_size = ', tmp_crop_size)
    #print('              reference_5pts = ', tmp_5pts)

    # 3) add outer_padding to make output_size
    reference_5point = tmp_5pts + np.array(outer_padding)
    tmp_crop_size = output_size
    #print('---> STEP3: add outer_padding to make output_size')
    #print('              crop_size = ', tmp_crop_size)
    #print('              reference_5pts = ', tmp_5pts)

    #print('===> end get_reference_facial_points\n')

    return reference_5point


def cropBox(img, cropbox, resizebox=(112, 112), margin=0):
    '''
    从img给出的区域cropbox,切割出图片，然后缩放resizebox,
     返回（切割后的图片，切割的区域）
    :param img: 
    :param cropbox: 
    :param resizebox: 
    :param margin: 
    :return: 
    '''
    imgW, imgH = img.size

    x1, y1, x2, y2 = cropbox[0], cropbox[1], cropbox[2], cropbox[3]
    w, h = resizebox

    padX = int(margin / 2 / w * imgW)
    padY = int(margin / 2 / h * imgH)

    x1 = max(0, x1 - padX)
    y1 = max(0, y1 - padY)
    x2 = min(imgW, x2 + padX)
    y2 = min(imgH, y2 + padY)
    cropBox = (x1, y1, x2, y2)


    r = img.crop((x1, y1, x2, y2)).resize(resizebox)
    return r, cropBox


def scaleLandmark(box, landmark, newbox=(112, 112)):
    x1, y1 = box[0], box[1]
    W, H = box[2] - box[0], box[3] - box[1]
    w, h = newbox

    r = np.zeros_like(landmark)
    for i, ll in enumerate(landmark):
        r[i] = landmark[i, 0] - x1, landmark[i, 1] - y1
        r[i] = r[i, 0] / W * w, r[i, 1] / H * h
    return r


def align_leastSquare(uv, xy):
    '''
    uv:(?,2)
    xy:(?,2)

    xy--> uv
    '''

    Y = np.concatenate([uv[:, 0], uv[:, 1]])
    x, y = xy[:, 0], xy[:, 1]
    p1 = np.stack([x, y, np.ones_like(x), np.zeros_like(x)], axis=1)
    p2 = np.stack([y, -x, np.zeros_like(x), np.ones_like(x)], axis=1)
    A = np.concatenate((p1, p2), axis=0)

    w, _, _, _ = np.linalg.lstsq(A, Y)
    M = np.array([[w[0], w[1], w[2]], [-w[1], w[0], w[3]]])
    return M


def flipLandMark(landmark,W=112):
    landmark_new=landmark.copy()
    landmark_new[:,0]=W-landmark_new[:,0]
    landmark_new[[0,1]]=landmark_new[[1,0]]
    landmark_new[[3,4]]=landmark_new[[4,3]]
    return landmark_new

class AlignFace():
    def __init__(self,img_size=(112,112)):
        self.ref=get_reference_facial_points(output_size=img_size,inner_padding_factor=1e-4,default_square=True)
        self.img_size=img_size

    def align(self,img,bbox,landmark,margin,flag=0):
        '''
        返回对齐后的图片，图片被缩放到了相应的"尺寸"了
        
        :param img: 
        :param bbox: 
        :param landmark: 
        :param margin: 
        :param flag: 
        :return: 
        '''
        img,bbox,landmark=self._crop_(img,bbox,landmark,margin)

        if flag==0:
            M=self._align_similar_none_reflective(landmark)
        if flag==1:
            M,needFlip=self._align_similar_reflective(landmark)
            if needFlip:
                img=img.transpose(Image.FLIP_LEFT_RIGHT)
        I=np.array(img)
        I=cv2.warpAffine(I,M,self.img_size)
        return Image.fromarray(I)

    def _crop_(self,img,bbox,landmark,margin):
        '''
        切割出 给定的 “bbox”,调整相应的landmark,
        返回新的 (图片,切割后的bbox,五官landmark)
        :param img: 
        :param bbox: 
        :param landmark: 
        :param margin: 
        :return: 
        '''
        img,bbox=cropBox(img,bbox,self.img_size,margin)
        landmark=scaleLandmark(bbox,landmark,self.img_size)
        return img,bbox,landmark

    def _align_similar_none_reflective(self,landmark):
        '''
        
        使用给出的lankmark与ref比较,获得变化矩阵M
        
        :param img: 
        :param bbox: 
        :param landmark: 
        :return: 
        '''
        M=align_leastSquare(uv=self.ref,xy=landmark)
        return M

    def _residual(self,landmark,M):
        R=M[:2,:2].T
        T=M[:,2]

        a=landmark.dot(R)+T
        b=self.ref
        return np.linalg.norm(a-b)

    def _align_similar_reflective(self,landmark):
        M1=self._align_similar_none_reflective(landmark)
        r1=self._residual(landmark,M1)

        landmark_fp=flipLandMark(landmark,self.img_size[0])
        M2=self._align_similar_none_reflective(landmark_fp)
        r2=self._residual(landmark_fp,M2)
        print(r1,r2)
        if r1>r2:
            return M2,True
        else:
            return M1,False
