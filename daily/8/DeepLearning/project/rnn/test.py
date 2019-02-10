import tensorflow as tf


N=71

for n in range(1000):
    if n%N==10:print(n)
# tf.add_to_collection('zxk',{1:2,3:3})
# tf.add_to_collection('zxk',{22:22,33:33})
# print(tf.get_collection_ref('zxk'))
# lr=tf.Variable(0.0,False)
# new_lr=tf.placeholder(tf.float32,[])
# update_lr=tf.assign(lr,new_lr)
#
# with tf.Session() as sess:
#     for i in range(10):
#         _lr=sess.run(update_lr,feed_dict={new_lr:i*1.0})
#         print(_lr)
# N=128
# DH=32
# Dx=500
# T=7
#
# def print_variable():
#     xx=tf.get_collection(tf.GraphKeys.VARIABLES)
#
#     print(xx)
# X=tf.placeholder(dtype=tf.float32,shape=[N,T,Dx])
# cell=tf.contrib.rnn.LSTMBlockCell(DH)
# print_variable()
# init_state=cell.zero_state(N,tf.float32)
# print_variable()
#
# for t in range(2):
#     _,init_state=cell(X[:,t,:],init_state)
# print_variable()
# print(init_state)
# W,b=cell.get_weights()
# print(W.shape)
# print(b.shape)
# print(b)
# print(cell.state_size)
# class hello():
#     def __init__(self):
#         self._age=10
#         self._mon='xx'
#     @property
#     def age(self):
#         return self._age
#     @property
#     def mon(self):
#         return self._mon
#
# h=hello()
# print(h.age)
# print(h.mon)
a={'hh/ss':1}
a.update(cc='ss',dd='fff')
print(a)