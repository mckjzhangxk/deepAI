import tensorflow as tf
import numpy as np


# g1=tf.Graph()
# g2=tf.Graph()
#
# with g1.as_default():
#     sess1=tf.Session()
#     model1=tf.random_normal(name='model1',shape=(4,))
#     print(sess1.run(model1))
# print('model1 from graph1:',g1.get_tensor_by_name('model1:0'))
# with g2.as_default():
#     sess2=tf.Session()
#     model2=tf.random_normal(name='model2',shape=(2,))
#     print(sess2.run(model2))
# print('model2 from graph2:',g2.get_tensor_by_name('model2:0'))
# print(sess2.run(g2.get_tensor_by_name('model2:0')))
# print(sess1.run(g1.get_tensor_by_name('model1:0')))
# # print(sess1.run(g2.get_tensor_by_name('model2:0')))
# print(sess1.graph)
# print(sess2.graph)
# print(g1)
# print(g2)


# a = tf.constant(1.0)
# b = tf.constant(2.0)
# mainSession=tf.Session()
# with mainSession.as_default():
#    print(tf.get_default_session())
#    print(a.eval())
# with tf.Session():
#    print(tf.get_default_session())
#    print(b.eval())
#
# print(mainSession.run(a))
# print(mainSession.run(b))
# print(tf.get_default_session())
import tensorflow.contrib.slim as slim

with tf.variable_scope('main'):
   I = slim.variable('Input', [100, 32, 32, 3])
   with slim.arg_scope([slim.conv2d],weights_initializer=tf.truncated_normal_initializer(stddev=.1),
                       weights_regularizer=slim.l2_regularizer(.1),
                       stride=2,padding='VALID'):
      # net_a=slim.conv2d(I,16,3)
      # net_b = slim.conv2d(I, 16, 3)
      # net_c = slim.conv2d(I, 16, 3)
      # print(net_a)
      # print(net_b)
      # print(net_c)
      # o=tf.concat([net_a,net_b,net_c],axis=3)
      # print(o)
      o=slim.repeat(I,3,slim.conv2d,num_outputs=22,kernel_size=3)
      print(o)