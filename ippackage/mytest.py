import tensorflow as tf
from data import TrainInput,get_input
from model.RnnModel import BaseModel as Model

hparam=tf.contrib.training.HParams(
    mode='train',
    rnn_type='lstm',
    ndims=128,
    num_layers=2,
    num_output=2,
    batch_size=256,
    dropout=0.2,
    forget_bias=1.0,
    residual=False,
    perodic=3,
    Tmax=6,     #序列的最大长度,是文件的宽度/特征数量
    lr=1e-4,
    solver='sgd',
    num_train_steps=12000,
    decay_scheme=None ,# "luong5", "luong10", "luong234"
    max_gradient_norm=5,
    features=2,

    src_path='/home/zhangxk/projects/deepAI/ippackage/data/data',
    scope='VPNNetWork',
)


path='/home/zhangxk/projects/deepAI/ippackage/data/data'
myinput=get_input(path,
          BATCH_SIZE=hparam.batch_size,
          Tmax=hparam.Tmax,
          D=hparam.features,
          perodic=hparam.perodic)
model=Model(myinput,hparam)
for v in tf.global_variables():
    print(v)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(model.reset_source_op)
    op=[model._loss,model.train_op,model.transfer_initState_op]
    op1=[model.accuracy,model.summary_op]+op

    for s in range(30):
        try:
            sess.run(model.feed_source_op)
            for i in range(2):
                    # print(sess.run(myinput.X))
                    # print(sess.run(myinput.batch_size))
                _loss,_,_,_xxx,_valid,_cursor=sess.run(op+[model.xxx,model._valid_idx,model._input.Cursor])
                print(_xxx)
                print(_valid)
                print(i,_cursor)
                print('---------------------')
            sess.run(model.reset_initState_op)
        except tf.errors.OutOfRangeError:
            sess.run(model.reset_source_op)
            print(s,'err')