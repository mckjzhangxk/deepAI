# -*- coding: utf-8 -*-
# File: resnet_model.py

import tensorflow as tf
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.models import BatchNorm, BNReLU, Conv2D, FullyConnected, GlobalAvgPooling, MaxPooling
from tensorpack.models import regularize_cost, l2_regularizer
from tensorpack.tfutils.argscope import argscope, get_arg_scope
from tensorpack.tfutils.tower import TowerContext
from tensorpack.utils import logger
from modelzoo.tensorflow.inputs import DataIterator
import numpy as np
from tqdm import tqdm
from common.utils import restore_from_npz


def resnet_shortcut(l, n_out, stride, activation=tf.identity):
    data_format = get_arg_scope()['Conv2D']['data_format']
    n_in = l.get_shape().as_list()[1 if data_format in ['NCHW', 'channels_first'] else 3]
    if n_in != n_out:   # change dimension when channel is not the same
        return Conv2D('convshortcut', l, n_out, 1, strides=stride, activation=activation)
    else:
        return l


def apply_preactivation(l, preact):
    if preact == 'bnrelu':
        shortcut = l    # preserve identity mapping
        l = BNReLU('preact', l)
    else:
        shortcut = l
    return l, shortcut


def get_bn(zero_init=False):
    """
    Zero init gamma is good for resnet. See https://arxiv.org/abs/1706.02677.
    """
    if zero_init:
        return lambda x, name=None: BatchNorm('bn', x, gamma_initializer=tf.zeros_initializer())
    else:
        return lambda x, name=None: BatchNorm('bn', x)


def preresnet_basicblock(l, ch_out, stride, preact):
    l, shortcut = apply_preactivation(l, preact)
    l = Conv2D('conv1', l, ch_out, 3, strides=stride, activation=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3)
    return l + resnet_shortcut(shortcut, ch_out, stride)


def preresnet_bottleneck(l, ch_out, stride, preact):
    # stride is applied on the second conv, following fb.resnet.torch
    l, shortcut = apply_preactivation(l, preact)
    l = Conv2D('conv1', l, ch_out, 1, activation=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, strides=stride, activation=BNReLU)
    l = Conv2D('conv3', l, ch_out * 4, 1)
    return l + resnet_shortcut(shortcut, ch_out * 4, stride)


def preresnet_group(name, l, block_func, features, count, stride):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                # first block doesn't need activation
                l = block_func(l, features,
                               stride if i == 0 else 1,
                               'no_preact' if i == 0 else 'bnrelu')
        # end of each group need an extra activation
        l = BNReLU('bnlast', l)
    return l


def resnet_basicblock(l, ch_out, stride):
    shortcut = l
    l = Conv2D('conv1', l, ch_out, 3, strides=stride, activation=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, activation=get_bn(zero_init=True))
    out = l + resnet_shortcut(shortcut, ch_out, stride, activation=get_bn(zero_init=False))
    return tf.nn.relu(out)


def resnet_bottleneck(l, ch_out, stride, stride_first=False):
    """
    stride_first: original resnet put stride on first conv. fb.resnet.torch put stride on second conv.
    """
    shortcut = l
    l = Conv2D('conv1', l, ch_out, 1, strides=stride if stride_first else 1, activation=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, strides=1 if stride_first else stride, activation=BNReLU)
    l = Conv2D('conv3', l, ch_out * 4, 1, activation=get_bn(zero_init=True))
    out = l + resnet_shortcut(shortcut, ch_out * 4, stride, activation=get_bn(zero_init=False))
    return tf.nn.relu(out)


def se_resnet_bottleneck(l, ch_out, stride):
    shortcut = l
    l = Conv2D('conv1', l, ch_out, 1, activation=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, strides=stride, activation=BNReLU)
    l = Conv2D('conv3', l, ch_out * 4, 1, activation=get_bn(zero_init=True))

    squeeze = GlobalAvgPooling('gap', l)
    squeeze = FullyConnected('fc1', squeeze, ch_out // 4, activation=tf.nn.relu)
    squeeze = FullyConnected('fc2', squeeze, ch_out * 4, activation=tf.nn.sigmoid)
    data_format = get_arg_scope()['Conv2D']['data_format']
    ch_ax = 1 if data_format in ['NCHW', 'channels_first'] else 3
    shape = [-1, 1, 1, 1]
    shape[ch_ax] = ch_out * 4
    l = l * tf.reshape(squeeze, shape)
    out = l + resnet_shortcut(shortcut, ch_out * 4, stride, activation=get_bn(zero_init=False))
    return tf.nn.relu(out)


def resnet_group(name, l, block_func, features, count, stride):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                l = block_func(l, features, stride if i == 0 else 1)
    return l


def resnet_backbone(image, num_blocks, group_func, block_func):
    with argscope(Conv2D, use_bias=False,
                  kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out')):
        # Note that this pads the image by [2, 3] instead of [3, 2].
        # Similar things happen in later stride=2 layers as well.
        l = Conv2D('conv0', image, 64, 7, strides=2, activation=BNReLU)
        l = MaxPooling('pool0', l, pool_size=3, strides=2, padding='SAME')
        l = group_func('group0', l, block_func, 64, num_blocks[0], 1)
        l = group_func('group1', l, block_func, 128, num_blocks[1], 2)
        l = group_func('group2', l, block_func, 256, num_blocks[2], 2)
        l = group_func('group3', l, block_func, 512, num_blocks[3], 2)
        l = GlobalAvgPooling('gap', l)
        logits = FullyConnected('linear', l, 1000,
                                kernel_initializer=tf.random_normal_initializer(stddev=0.01))
    return logits




class ImageNetModel():
    image_shape = 224

    """
    uint8 instead of float32 is used as input type to reduce copy overhead.
    It might hurt the performance a liiiitle bit.
    The pretrained models were trained with float32.
    """
    image_dtype = tf.uint8

    """
    Either 'NCHW' or 'NHWC'
    """
    data_format = 'NCHW'

    """
    Whether the image is BGR or RGB. If using DataFlow, then it should be BGR.
    """
    image_bgr = True

    weight_decay = 1e-4

    """
    To apply on normalization parameters, use '.*/W|.*/gamma|.*/beta'
    """
    weight_decay_pattern = '.*/W'

    """
    Scale the loss, for whatever reasons (e.g., gradient averaging, fp16 training, etc)
    """
    loss_scale = 1.

    """
    Label smoothing (See tf.losses.softmax_cross_entropy)
    """
    label_smoothing = 0.

    def forward(self, inputs, is_training=False):
        image = self.image_preprocess(inputs)
        assert self.data_format in ['NCHW', 'NHWC']
        if self.data_format == 'NCHW':
            image = tf.transpose(image, [0, 3, 1, 2])
        with TowerContext('', is_training):
            logits = self.get_logits(image)
        return logits

    def get_logits(self, image):
        """
        Args:
            image: 4D tensor of ``self.input_shape`` in ``self.data_format``

        Returns:
            Nx#class logits
        """
        pass

    def image_preprocess(self, image):
        with tf.name_scope('image_preprocess'):
            if image.dtype.base_dtype != tf.float32:
                image = tf.cast(image, tf.float32)
            mean = [0.485, 0.456, 0.406]    # rgb
            std = [0.229, 0.224, 0.225]
            if self.image_bgr:
                mean = mean[::-1]
                std = std[::-1]
            image_mean = tf.constant(mean, dtype=tf.float32) * 255.
            image_std = tf.constant(std, dtype=tf.float32) * 255.
            image = (image - image_mean) / image_std
            return image

    def compute_loss_and_error(self,logits, label, label_smoothing=0.):
        if label_smoothing != 0.:
            nclass = logits.shape[-1]
            label = tf.one_hot(label, nclass) if label.shape.ndims == 1 else label

        if label.shape.ndims == 1:
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        else:
            loss = tf.losses.softmax_cross_entropy(
                label, logits, label_smoothing=label_smoothing,
                reduction=tf.losses.Reduction.NONE)
        loss = tf.reduce_mean(loss, name='xentropy-loss')


        if self.weight_decay > 0:
            wd_loss = regularize_cost(self.weight_decay_pattern,
                                      l2_regularizer(self.weight_decay),
                                      name='l2_regularize_loss')
            add_moving_summary(loss, wd_loss)
            total_cost = tf.add_n([loss, wd_loss], name='cost')
        else:
            total_cost = tf.identity(loss, name='cost')
            add_moving_summary(total_cost)

        if self.loss_scale != 1.:
            logger.info("Scaling the total loss by {} ...".format(self.loss_scale))

        def prediction_incorrect(logits, label, topk=1, name='incorrect_vector'):
            with tf.name_scope('prediction_incorrect'):
                x = tf.logical_not(tf.nn.in_top_k(logits, label, topk))
            return tf.cast(x, tf.float32, name=name)

        wrong = prediction_incorrect(logits, label, 1, name='wrong-top1')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top1'))

        wrong = prediction_incorrect(logits, label, 5, name='wrong-top5')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top5'))
        return total_cost

class Model(ImageNetModel):
    def __init__(self, depth, mode='resnet'):
        if mode == 'se':
            assert depth >= 50

        self.mode = mode
        basicblock = preresnet_basicblock if mode == 'preact' else resnet_basicblock
        bottleneck = {
            'resnet': resnet_bottleneck,
            'preact': preresnet_bottleneck,
            'se': se_resnet_bottleneck}[mode]
        self.num_blocks, self.block_func = {
            18: ([2, 2, 2, 2], basicblock),
            34: ([3, 4, 6, 3], basicblock),
            50: ([3, 4, 6, 3], bottleneck),
            101: ([3, 4, 23, 3], bottleneck),
            152: ([3, 8, 36, 3], bottleneck)
        }[depth]

    def get_logits(self, image):
        with argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm], data_format=self.data_format):
            return resnet_backbone(
                image, self.num_blocks,
                preresnet_group if self.mode == 'preact' else resnet_group, self.block_func)
class ResNet50(Model):
    def __init__(self, model_path=None):
        super().__init__(50,'resnet')
        self._build_()
        if model_path:
            self.session=tf.Session()
            self.restore_model(model_path)
    def _build_(self):
        self.X = tf.placeholder(tf.uint8, shape=[None, 224, 224, 3])
        self.logit=self.forward(self.X, False)

    def restore_model(self,path):
        restore_from_npz(self.session,path)

    def _build_eval_graph(self):
        self.Y= tf.placeholder(tf.int32, shape=[None])
        self.top1=  tf.nn.in_top_k(self.logit, self.Y, 1)
        self.top5 = tf.nn.in_top_k(self.logit, self.Y, 5)
    def evaluate(self,evalfile,prefix='',batchsize=32):
        self._build_eval_graph()
        ds=DataIterator(prefix,evalfile,batchsize)
        C1,C2,N=0,0,0
        for x,y in tqdm(ds,'Doing Evaluation:'):
            c1,c2=self.session.run([self.top1,self.top5],feed_dict={self.X:x,self.Y:y})
            C1 += c1
            C2 += c2
            N+=len(x)
        top1,top5=c1/N,c2/N
        return top1,top5

if __name__ == '__main__':
    model=ResNet50('/home/zhangxk/桌面/weight/ImageNet-ResNet50.npz')
    # top1,top5=model.evaluate('','','')
