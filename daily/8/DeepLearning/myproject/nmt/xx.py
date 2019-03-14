import tensorflow as tf
import collections
# global_step=tf.Variable(0,trainable=False,name='global_step')
# ass=tf.assign(global_step,global_step+1)
#
# cc=tf.cond(global_step<5,lambda:tf.constant(0),lambda :global_step)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for i in range(10):
#         sess.run(ass)
#         # print(sess.run(global_step))
#         print(sess.run(cc))

# D1=collections.Counter()
# D2=collections.Counter()
#
# D1['a']=1
# D1['b']=2
# D1['c']=1
#
#
# D2['d']=1
# D2['b']=1
# D2['c']=1
#
# print(D1&D2)
# print(D1|D2)

import codecs
translation=['Rachel Pike : The science behind a climate headline','Khoa học đằng sau một tiêu đề về khí hậu']
with codecs.open('xx', mode='w', encoding='utf-8') as fs:
    for t in translation:
        fs.write(t + '\n')