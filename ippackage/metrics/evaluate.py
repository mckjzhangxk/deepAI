import tensorflow as tf

def accuracy(logit, label, valid_idx=None):
    '''
    logit:是一个(batch,C)的tensor
    label:(batch,)标注
    valid:(batch,)如果是None,默认都是有效的

    返回准确率
    :param logit: 
    :param label: 
    :param valid_idx: 
    :return: 
    '''

    if valid_idx is None:
        valid_idx = tf.to_float(tf.ones_like(logit))
    else:
        valid_idx = tf.to_float(valid_idx)

    predict = tf.argmax(logit,axis=1, output_type=tf.int32)
    true_predict = tf.to_float(tf.equal(predict, label))
    true_predict_cnt = tf.reduce_sum(true_predict * valid_idx)
    cnt = tf.reduce_sum(valid_idx)
    return tf.divide(true_predict_cnt, cnt+ 1e-13)


def accuracyPerClass(logit,label,valid_idx=None,C=2):
    '''
    logit:是一个(batch,C)的tensor
    label:(batch,)标注
    valid:(batch,)如果是None,默认都是有效的
    C:类型数量
    返回一个list,元素是针对每个类的accuracy(shape(),float32)
    :param logit: 
    :param label: 
    :param valid_idx: 
    :param C: 
    :return: 
    '''

    if valid_idx is None:
        valid_idx=tf.ones_like(label)
    valid_idx=tf.to_int32(valid_idx)

    predict=tf.argmax(logit,axis=1,output_type=tf.int32)
    res=[]
    for c in range(C):
        label_c  =tf.to_int32(tf.equal(label, c))*valid_idx
        predict_c=tf.to_int32(tf.equal(predict, c))*valid_idx


        predict_right=tf.to_float(
            tf.logical_and(
                tf.cast(predict_c,tf.bool),
                tf.cast(label_c, tf.bool)
                           )
        )
        #label=c and predict=2的个数
        true_predict_cnt=tf.to_float(tf.reduce_sum(predict_right))
        #label==c的个数
        gt_true_cnt=tf.to_float(tf.reduce_sum(label_c))
        res.append(tf.divide(true_predict_cnt,gt_true_cnt+1e-8,name='class_%d_acc'%c))

    return res