import tensorflow as tf
import tensorflow.contrib as tfcontrib
import functools
import numpy as np
from utils.iterator_utils import prepare_dataset
conv3_3=functools.partial(tf.layers.Conv2D,
                  kernel_size=(3,3),
                  strides=(1,1),
                  padding='same',
                  data_format='channels_last',
                  use_bias=True,
                  kernel_initializer=tfcontrib.layers.xavier_initializer_conv2d(),
                  bias_initializer=tf.zeros_initializer())

conv1_1=functools.partial(tf.layers.Conv2D,
                  kernel_size=(1,1),
                  strides=(1,1),
                  padding='same',
                  data_format='channels_last',
                  use_bias=True,
                  kernel_initializer=tfcontrib.layers.xavier_initializer_conv2d(),
                  bias_initializer=tf.zeros_initializer())
max_pool=functools.partial(
            tf.layers.MaxPooling2D,
            pool_size=(2,2),
            strides=(2,2),
            padding='same',
            data_format='channels_last'
)()
bn=functools.partial(
    tf.layers.BatchNormalization,
    axis=-1,
    momentum=0.99,
    center=True,
    scale=True,
)
up_conv=functools.partial(tf.layers.Conv2DTranspose,
                          kernel_size=(3,3),
                          strides=(2,2),
                          padding='same',
                          use_bias=True,
                          kernel_initializer=tfcontrib.layers.xavier_initializer_conv2d(),
                          bias_initializer=tf.zeros_initializer()
                          )
concat=functools.partial(tf.concat,axis=-1)
relu=functools.partial(tf.nn.relu)

def dice_loss(labels, logits, smooth):
    predict=tf.nn.sigmoid(logits)
    _infersect=tf.reduce_sum(labels * predict)
    a=2.0*_infersect+smooth
    b= tf.reduce_sum(labels) + tf.reduce_sum(predict) + smooth

    return 1-a/b
class Unet():

    def __init__(self,hparam,batch_input,runMode):
        self._batch_input=batch_input
        self.runMode =runMode
        self.mode=True if runMode=='train' else False
        self._stack=[]
        self._buildGraph(hparam)
    def _buildGraph(self,hparam):
        self._build_encoder(hparam)
        self._build_decoder(hparam)

        if self.runMode=='train':
            self._set_train(hparam)
            self._set_train_summary()
        if self.runMode == 'eval':
            self._set_eval(hparam)
            self._set_eval_summary()
        self._saver=tf.train.Saver(tf.trainable_variables(),max_to_keep=hparam.max_keep)

    def _build_encoder(self,hparam):
        _next=self._batch_input.X

        with tf.variable_scope('encoder',reuse=tf.AUTO_REUSE):
            for i in range(5):
                if i!=0:
                    current,_next=self._encode_block(_next,'layer_%d'%(i+1))
                else:
                    current, _next = self._encode_block(_next, 'layer_%d' % (i + 1),32)
                self._stack.append(current)
            self._center=self._conv_block(_next)

    def _build_decoder(self,hparam):
        _prev=self._center
        with tf.variable_scope('decoder',reuse=tf.AUTO_REUSE):
            for i in range(4,-1,-1):
                _prev=self._decode_block(self._stack[i],_prev,'layer_%d'%(i+1))

            self._logit=conv1_1(1)(_prev)

    def _conv_block(self,inp,dim=None):
        X=inp
        ndim=inp.get_shape()[-1]*2 if dim==None else dim
        for i in range(2):
            X=conv3_3(ndim)(X)
            X=bn()(X,self.mode)
            X=relu(X)
        return X

    def _encode_block(self, inp,name='',dim=None):
        '''
        inp:(?,H,W,C)
        输出:
        a:(?, H, W,  2C)
        b:(?,H/2,W/2,2C)
        :param inp: 
        :return: 
        '''
        with tf.variable_scope(name):
            a=self._conv_block(inp,dim)
            b=max_pool(a)
        return a,b

    def _decode_block(self,right_inp,bottom_inp,name=''):
        '''
        right:(H,W,C)
        bottom:(H/2,W/2,2*C)
        输出:
        (H,W,C)
        
        :param right_inp: 
        :param bottom_inp: 
        :return: 
        '''

        ndim=right_inp.get_shape()[-1]


        with tf.variable_scope(name):
            a=up_conv(ndim)(bottom_inp)
            X=concat([a,right_inp])
            X=bn()(X,self.mode)
            X=relu(X)
            for i in  range(2):
                X=conv3_3(ndim)(X)
                X = bn()(X, self.mode)
                X = relu(X)
            return X

    def _set_loss(self,smooth):

        self._binary_loss=tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self._batch_input.Y,
                logits=self._logit)
        )
        self._dice_loss=dice_loss(
            labels=self._batch_input.Y,
            logits=self._logit,smooth=smooth)
        self._loss=self._binary_loss+self._dice_loss
    def _set_train(self,hparam):
        self._set_loss(hparam.smooth)
        lr=hparam.lr
        sovler=hparam.solver
        if sovler=='adam':
            optimizer=tf.train.AdamOptimizer(lr)
        elif sovler=='sgd':
            optimizer=tf.train.GradientDescentOptimizer(lr)
        elif sovler=='rmsp':
            optimizer=tf.train.RMSPropOptimizer(lr)
        self.global_step=tf.Variable(0,trainable=False,dtype=tf.int32)
        self._train_op=optimizer.minimize(self._loss,self.global_step)
    def _set_train_summary(self):
        vars=tf.trainable_variables()
        grads=tf.gradients(self._loss,vars)
        grad_norm=tf.global_norm(grads)

        tf.summary.scalar('grad_norm', grad_norm)
        tf.summary.scalar('loss', self._loss)
        tf.summary.scalar('binary_loss', self._binary_loss)
        tf.summary.scalar('dice_loss',self._dice_loss)

        self._train_summary=tf.summary.merge_all()
    def _set_eval(self,hparam):
        pred=tf.greater(self._logit , 0.0)
        label=tf.cast(self._batch_input.Y,tf.bool)

        true_predict=tf.reduce_sum(tf.to_float(tf.equal(pred,label)))
        total=tf.to_float(tf.size(label))
        self._accuracy=true_predict/total
    def _set_eval_summary(self):
        tf.summary.scalar('accuracy', self._accuracy)
        self._eval_summary = tf.summary.merge_all()

    def init(self,sess):
        sess.run(self._batch_input.initializer)
    def train(self,sess):
        op=[self._train_op,self.global_step,self._loss,self._train_summary]
        _,_global_step,_loss,_summary=sess.run(op)
        return (_global_step,_loss,_summary)
    def eval(self,sess):
        op = [self._accuracy,self._eval_summary]
        acc=[]
        self.init(sess)
        while True:
            try:
                _acc,_summary=sess.run(op)
                acc.append(_acc)
            except tf.errors.OutOfRangeError:
                break
        return np.average(acc),_summary

    def infer(self,sess):
        pass
    def save(self,sess,model_path,global_step):
        self._saver.save(sess,model_path,global_step)
# X=tf.random_uniform(shape=(55,256,256,96))
# T=bn(X,False)
# print(T)
# print(concat([X,X,X]))
# for f in tf.global_variables():
#     print(f)
# print(up_conv(96)(X))
# Y=conv1_1(1)(X)
# print(max_pool()(Y))
# print(Y)

# hparam=tfcontrib.training.HParams(
#     src_prefix='/home/jncf/segment/train',
#     tgt_prefix='/home/jncf/segment/train_masks',
#     train_mask_file='/home/jncf/segment/train_masks_1.csv',
#     num_parallel_calls=4,
#     batch_size=128,
#     epoch=3,
#     train_size=0.8,
#     shift_range=0.1,
#     hue_delta=0.1,
#     image_size=(256,256),
#     scale=1.0,
#
#     lr=1e-3,
#     solver='adam',
#     smooth=1,
# )
# X=tf.random_uniform(shape=(12,24,24,3))
# Y=tf.random_uniform(shape=(12,24,24,3))
# print(bn()(X))
# for v in tf.global_variables():
#     print(v)
# print(bn()(Y))
# for v in tf.global_variables():
#     print(v)
# train_batch,test_batch=prepare_dataset(hparam)
#
# model=Unet(hparam,train_batch,'train')
# model1=Unet(hparam,test_batch,'eval')
#
# for v in tf.trainable_variables():
#     print(v)