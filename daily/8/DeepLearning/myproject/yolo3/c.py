import numpy as np
import numpy.random as npr
import tensorflow as tf

g=tf.Graph()
with g.as_default():
    with tf.Session() as sess:
        X=tf.placeholder(tf.float32,shape=())
        with tf.variable_scope('scope1'):
            W1=tf.get_variable('W1',shape=())
            Y=X*W1
        out1=[Y.name[:-2]]


        with tf.variable_scope('scope2'):
            W2=tf.get_variable('W2',shape=())
            Z=X*W2
        out2=[Z.name[:-2]]

        print(X.name)
        print(Y.name)
        print(Z.name)

        sess.run(tf.global_variables_initializer())
        g1_def = tf.graph_util.convert_variables_to_constants(sess, g.as_graph_def(), out1)
        g2_def = tf.graph_util.convert_variables_to_constants(sess, g.as_graph_def(), out2)

        with tf.gfile.GFile('g1.pb','wb') as f1:
            f1.write(g1_def.SerializeToString())
        with tf.gfile.GFile('g2.pb', 'wb') as f2:
            f2.write(g2_def.SerializeToString())
        # f1=tf.summary.FileWriter('log1',g)
        # print(len(g1_def.node))
        # print(len(g2_def.node))
        # f1.close()
        # f2=tf.summary.FileWriter('log2',graph_def=g2_def)
        # f2.close()