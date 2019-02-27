import numpy as np
import tensorflow as tf
import collections


def _get_cursor(p):
    '''
    P:int,cursor每次增长量
    创建变量cursor保存当前位置,和更新操作pointer
    
    每次调用pointer,cursor都会+p
    返回(pointor,cursor)
    :param p: 
    :return: 
    '''
    with tf.variable_scope('Input'):
        cursor=tf.get_variable('cursor',shape=(),dtype=tf.int32,trainable=False
                           ,initializer=tf.constant_initializer(-p))
        pointor=tf.assign(cursor,(cursor+p)%12,name='pointor')
        return pointor,cursor

P=5
pointor,cursor=_get_cursor(P)

a=tf.get_variable('a',shape=[12])
b=a[pointor:pointor+P]

sess=tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(a.assign(np.arange(12)))
for i in range(5):
    # print(sess.run(pointor))
    print(sess.run(b))
    print(sess.run(cursor))
    # print(sess.run(cursor))
    # print(sess.run(cursor))
sess.close()
