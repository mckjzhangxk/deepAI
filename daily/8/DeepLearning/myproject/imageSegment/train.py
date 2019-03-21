import tensorflow as tf
import tensorflow.contrib as tfcontrib
from utils.iterator_utils import prepare_dataset
from model import Unet
import time
import numpy as np

hparam=tfcontrib.training.HParams(
    src_prefix='/home/zxk/AI/carana/carvana-image-masking-challenge/train',
    tgt_prefix='/home/zxk/AI/carana/carvana-image-masking-challenge/train_masks',
    train_mask_file='/home/zxk/AI/carana/carvana-image-masking-challenge/train_masks.csv',
    num_parallel_calls=4,
    batch_size=1,
    epoch=30,
    train_size=0.8,
    shift_range=0.1,
    hue_delta=0.1,
    image_size=(256,256),
    scale=1/255,

    lr=1e-3,
    solver='adam',
    smooth=1,
    log_dir='model/log',
    steps_per_state=10,
    steps_per_eval=2000,
    model_path='model/UNet',
    max_keep=5
)


def createModel(hparam):
    graph=tf.Graph()
    with graph.as_default():
        train_input, test_input = prepare_dataset(hparam)
        sess=tf.Session()
        train_model=Unet(hparam,train_input,'train')
        eval_model=Unet(hparam,test_input,'eval')
    return (graph,sess,train_model,eval_model)


def train(hparam):
    '''
    创建 train,eval两个计算图,读取数据集进行训练,数据集结束的
    时候eval,统计准确性.
    每steps_per_state打印在这个区间内计算的平均loss,accuarcy
    
    :param hparam: 
    :return: 
    '''
    graph, sess, train_model, eval_model=createModel(hparam)


    with sess:
        with tf.summary.FileWriter(hparam.log_dir,graph) as logfs:
            sess.run(tf.global_variables_initializer())
            train_model.init(sess)
            start_time=time.time()
            while True:
                try:
                    global_steps,_loss,_summary=train_model.train(sess)
                    if global_steps % hparam.steps_per_state == 0:
                        logfs.add_summary(_summary, global_steps)
                        endtime=time.time()
                        speed=hparam.steps_per_state/(endtime-start_time)
                        print('step %d,loss is %f,speed is %f ps'%(global_steps,_loss,speed))
                        start_time = time.time()
                    if global_steps % hparam.steps_per_eval == 0:
                        eval_model.init(sess)
                        _acc,_summary=eval_model.eval(sess)
                        logfs.add_summary(_summary, global_steps)
                        print('Eval,Accuaray is %f'%_acc)
                        eval_model.save(sess,hparam.model_path,global_steps)

                except tf.errors.OutOfRangeError:
                    print('finish training')
                    break

train(hparam)
