import tensorflow as tf
import sys
import numpy as np
slim = tf.contrib.slim

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

def createFeature(keys,values):
    values=[tf.train.Feature(bytes_list=tf.train.BytesList(value=[v])) for v in values]

    features={keys[i]:values[i] for i in range(len(keys))}
    example=tf.train.Example(features=tf.train.Features(feature=features))
    return example

class darknet53(object):
    """network for performing feature extraction"""

    def __init__(self, inputs):
        '''
        
        :param inputs: 来自Dataset的Image,shape(?,H,W,3)
        '''
        self.inputs=inputs
        batch_norm_params = {
            'epsilon': 1e-05,
            'scale': True,
            'is_training': False,
            'fused': None,  # Use fused batch norm if possible.
        }
        with tf.variable_scope('yolov3'):
            with slim.arg_scope([slim.conv2d],
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                biases_initializer=None,
                                activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.1)):
                with tf.variable_scope('darknet-53'):
                    self.outputs = self.forward(inputs)
    def _darknet53_block(self, inputs, filters):
        """
        implement residuals block in darknet53
        """
        shortcut = inputs
        inputs = _conv2d_fixed_padding(inputs, filters * 1, 1)
        inputs = _conv2d_fixed_padding(inputs, filters * 2, 3)

        inputs = inputs + shortcut
        return inputs

    def forward(self, inputs):

        inputs = _conv2d_fixed_padding(inputs, 32,  3, strides=1)
        inputs = _conv2d_fixed_padding(inputs, 64,  3, strides=2)
        inputs = self._darknet53_block(inputs, 32)
        inputs = _conv2d_fixed_padding(inputs, 128, 3, strides=2)

        for i in range(2):
            inputs = self._darknet53_block(inputs, 64)

        inputs = _conv2d_fixed_padding(inputs, 256, 3, strides=2)

        for i in range(8):
            inputs = self._darknet53_block(inputs, 128)

        route_1 = inputs
        inputs = _conv2d_fixed_padding(inputs, 512, 3, strides=2)

        for i in range(8):
            inputs = self._darknet53_block(inputs, 256)

        route_2 = inputs
        inputs = _conv2d_fixed_padding(inputs, 1024, 3, strides=2)

        for i in range(4):
            inputs = self._darknet53_block(inputs, 512)

        return route_1, route_2, inputs

    def extractFeature(self,model_file,tfrecord_file,dtype=np.float16):
        '''
        
        :param model_file: 模型路径
        :param tfrecord_file: 保存路径
        :return: 
        '''
        record_writer=tf.python_io.TFRecordWriter(tfrecord_file)
        with tf.Session() as sess:
            saver=tf.train.Saver()
            saver.restore(sess,model_file)

            cnt=0
            cache=[]
            try:
                while True:
                    y3,y2,y1=sess.run(self.outputs)
                    y1,y2,y3=dtype(y1),dtype(y2),dtype(y3)
                    n =len(y1)
                    for i in range(n):
                        example=createFeature(('y1','y2','y3'),(y1[i],y2[i],y3[i]))
                        cache.append(example.SerializeToString())
                    cnt+=1
                    if cnt%10==0:
                        progess_print('finish %d example'%(cnt*n))
                        for ex in cache:
                            record_writer.write(ex)
                        cache=[]

            except tf.errors.OutOfRangeError:pass
        record_writer.close()

