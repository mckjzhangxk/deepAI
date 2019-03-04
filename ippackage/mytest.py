import tensorflow as tf
from data import TrainInput,get_input
from model.RnnModel import BaseModel as Model

hparam=tf.contrib.training.HParams(
    mode='train',
    rnn_type='lstm',
    ndims=128,
    num_layers=2,
    num_output=2,
    batch_size=1,
    dropout=0.0,
    forget_bias=1.0,
    residual=False,
    perodic=3,
    Tmax=6,     #序列的最大长度,是文件的宽度/特征数量
    lr=1e-4,
    solver='adam',
    num_train_steps=100,
    decay_scheme=None ,# "luong5", "luong10", "luong234"
    max_gradient_norm=5,
    features=2,
    max_keeps=None,
    src_path='/home/zhangxk/projects/deepAI/ippackage/data/data1',
    scope='VPNNetWork',
)


path=hparam.src_path
myinput=get_input(path,
          BATCH_SIZE=hparam.batch_size,
          Tmax=hparam.Tmax,
          D=hparam.features,
          perodic=hparam.perodic)
model=Model(myinput,hparam,'train')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(model.reset_source_op)
    # op=[model._loss,model.train_op]
    feed_input_op=[model.feed_source_op,model.reset_initState_op]
    # sess.run(model.feed_source_op)
    for s in range(500):
        try:
            sess.run(feed_input_op)

            for i in range(2):

                _,_logit, _ = sess.run([model.train_op,
                                        model._logit,
                                        model.transfer_initState_op])
                # print('logit:',_logit)
                # print('cursor',sess.run(myinput.Cursor))
            print('-------------------')
        except tf.errors.OutOfRangeError:
            sess.run(model.reset_source_op)

    print('ttttttttttttttttttttttttttttttttttttttt')

    c=0
    acc1=[]
    acc2=[]
    acc=acc1

    sess.run(model.reset_source_op)
    while True:
        try:
            sess.run(feed_input_op)
            for i in range(2):
                _acc,_=sess.run([model.accuracy,model.transfer_initState_op])
                acc.append(_acc)
            print('-------------------')
        except tf.errors.OutOfRangeError:
            if c>=1:
                break
            sess.run(model.reset_source_op)
            c+=1
            acc=acc2
    print('xxxxxxxxxxxxxxxxxx')
    print(len(acc1))
    print(len(acc2))
    print(acc1)
    print(acc2)

    flag=True
    for a,b in zip(acc1,acc2):
        if a!=b:flag=False
    print(flag)
