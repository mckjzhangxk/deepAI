import tensorflow as tf
import sys
import cv2
slim = tf.contrib.slim

def processImage(impath,newshape=(416,416)):
    if isinstance(impath,str):
        im=cv2.imread(impath)
    else:im=impath

    oldH,oldW=im.shape[:2]
    im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    im=cv2.resize(im,(newshape[1],newshape[0]))
    im=im/255.0
    return im,(oldW,oldH)

def progess_print(info):
    sys.stdout.write('\r>>' + info)
    sys.stdout.flush()

def _conv2d_fixed_padding(inputs, filters, kernel_size, strides=1):
    if strides > 1: inputs = _fixed_padding(inputs, kernel_size)
    inputs = slim.conv2d(inputs, filters, kernel_size, stride=strides,
                         padding=('SAME' if strides == 1 else 'VALID'))
    return inputs


@tf.contrib.framework.add_arg_scope
def _fixed_padding(inputs, kernel_size, *args, mode='CONSTANT', **kwargs):
    """
    Pads the input along the spatial dimensions independently of input size.

    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                   Should be a positive integer.
      mode: The mode for tf.pad.

    Returns:
      A tensor with the same format as the input with the data either intact
      (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]], mode=mode)
    return padded_inputs