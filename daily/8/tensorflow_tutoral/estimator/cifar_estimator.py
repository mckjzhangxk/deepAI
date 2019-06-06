import tensorflow as tf
from CIFAR10Utils import load_dataset


def wrap_input_func(ds,batch_size=128,epoch=10):
    X_train, Y_train=ds
    def mycifarfunc():
        ds=tf.data.Dataset.from_tensor_slices(({'image':X_train},Y_train))
        return ds.repeat(-1).shuffle(batch_size*4).batch(batch_size)
    return mycifarfunc
def model_fn(features,labels,mode,params):

    X=features['image']/255.


    X=tf.layers.Conv2D(32,(3,3),padding='same',activation=tf.nn.relu)(X)
    X=tf.layers.max_pooling2d(X,(2,2),(2,2),padding='same')
    X=tf.layers.Conv2D(64,(3,3),padding='same',activation=tf.nn.relu)(X)
    X = tf.layers.max_pooling2d(X, (2, 2), (2, 2), padding='same')

    X=tf.reshape(X,(-1,8*8*64))

    logit=tf.layers.Dense(10,activation=None)(X)
    loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit,labels=labels))

    if mode==tf.estimator.ModeKeys.TRAIN:
        optimizer=tf.train.AdamOptimizer(1e-3)
        train_op=optimizer.minimize(loss,global_step=tf.train.get_global_step())
        tf.summary.scalar('loss',loss)
        return tf.estimator.EstimatorSpec(mode,loss=loss,train_op=train_op)
    if mode==tf.estimator.ModeKeys.EVAL:
        accuracy=tf.metrics.accuracy(labels=labels,predictions=tf.argmax(logit,1))
        metrics={'myaccuracy':accuracy}
        return tf.estimator.EstimatorSpec(mode,loss=loss,eval_metric_ops=metrics)
    if mode==tf.estimator.ModeKeys.PREDICT:
        classes=tf.argmax(logit,axis=1)
        prob=tf.nn.softmax(logit,axis=1)
        N=tf.shape(prob)[0]

        prob=tf.gather_nd(prob,
                          tf.stack((tf.range(N), tf.to_int32(classes)), axis=1)
                          )

        pred_dict={'classes':classes,'prob':prob}
        return tf.estimator.EstimatorSpec(mode,predictions=pred_dict)

if __name__ == '__main__':
    path='/home/zxk/AI/data/cifar/CIFAR10_DATA'
    train_x,train_y,test_x,test_y,_=load_dataset(flaten=False,one_hot=False,filename=path)
    tf.enable_eager_execution()
    train_inp_fn=wrap_input_func((train_x,train_y),batch_size=128)

    myestimator=tf.estimator.Estimator(model_fn=model_fn,model_dir='models',params={})
    myestimator.train(train_inp_fn,steps=5000)

    print('do evaluation....')
    test_inp_fn = wrap_input_func((test_x, test_y), batch_size=128)
    eval_result=myestimator.evaluate(test_inp_fn)
    print('eval_result:',eval_result)