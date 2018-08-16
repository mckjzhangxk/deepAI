import tensorflow as tf
import numpy as np

N,gridsize,C=256,3,7
# Y_orgin=tf.placeholder(dtype=tf.float32,shape=[None,gridsize,gridsize,C])
# Yhat_orgin=tf.placeholder(dtype=tf.float32,shape=[None,gridsize,gridsize,C])
#
#
# Y=tf.reshape(Y_orgin,[-1,C])
# Yhat=tf.reshape(Yhat_orgin,[-1,C])
#
#
# L1=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Yhat[:,0],labels=Y[:,0]))
#
#
# #mean square loss
# mark=tf.cast(tf.equal(Y[:,0:1],1),tf.float32)
# L2=tf.reduce_mean(mark*((Y[:,1:]-Yhat[:,1:])**2))
#
# Loss=L1+L2
# np.random.seed(0)
# y_data=np.random.rand(N,gridsize,gridsize,C)
# y_data[:,:,:,0]=np.random.randint(0,2,[N,gridsize,gridsize])
# y_hat_data=np.random.rand(N,gridsize,gridsize,C)
# y_hat_data[:,:,:,0]=y_data[:,:,:,0]*10000+(y_data[:,:,:,0]==0)*-1000
#
# with tf.Session() as sess:
#     _l1,_l2,_loss=sess.run([L1,L2,Loss],feed_dict={Y_orgin:y_data,Yhat_orgin:y_hat_data})
#     print(_l1)
#     print(_l2)
#     print(_loss)

numAnchors,classes=5,30
np.random.seed(1)

box_confidence=tf.Variable(initial_value=np.random.rand(gridsize,gridsize,numAnchors,1))
boxes=tf.Variable(initial_value=np.random.randn(gridsize,gridsize,numAnchors,4))
boxes_reshape=tf.reshape(boxes,[gridsize,gridsize,-1])
box_class_prob=tf.Variable(initial_value=np.random.rand(gridsize,gridsize,numAnchors,classes))


#the score for every box
box_score=tf.multiply(box_confidence,box_class_prob)
box_score_reshape=tf.reshape(box_score,[gridsize,gridsize,-1])


#max score for every box
box_max_score=tf.reduce_max(box_score,axis=(2,3))


box_argmax_score=tf.argmax(box_score_reshape,axis=2)
box_classes=tf.mod(box_argmax_score,classes)

cc=tf.gather(boxes_reshape,box_argmax_score)
print(cc)
mask=tf.greater(box_max_score,0.7)


# mask=np.random.rand(gridsize,gridsize)>0.5
# print(mask)

select_score=tf.boolean_mask(box_max_score,mask)
select_classes=tf.boolean_mask(box_classes,mask)
select_boxes=tf.boolean_mask(boxes_reshape,mask)
print(select_boxes)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    _mask,_classes,_score=sess.run([mask,select_classes,select_score])
    print(_mask)
    print()

    print(_classes)
    print()

    print(_score)
    print()