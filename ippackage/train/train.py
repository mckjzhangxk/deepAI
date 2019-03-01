import tensorflow as tf
import model.modelHelper as helper
from model import BaseModel
import time
from utils import print_state_info

def _initStatsInfo(hparam):
    start_time=time.time()

    return {
        'start_time':start_time,
        'total_loss':0.0,
        'accuracy':0.0,
        'learn_rate':0.0,
        'stat_steps':hparam.steps_per_state
    }
def _update_stat_info(stat_info,loss,accuracy,lr=None,globalstep=None):
    stat_info['total_loss']+=loss
    stat_info['accuracy'] += accuracy

    if globalstep:
        stat_info['steps']=globalstep
    if lr:
        stat_info['learn_rate']=lr

def train(hparam):
    '''
    
    :param hparam: 
    :return: 
    '''
    trainModel=helper.createTrainModel(hparam,BaseModel)
    model=trainModel.model

    stats_info=_initStatsInfo(hparam)


    common_op=[model.train_op,model.loss,model.accuracy,model.transfer_initState_op]
    stats_op =common_op+ [model.learning_rate,model.summary_op]

    with trainModel.graph.as_default() as graph:
        with trainModel.session as sess:
            with tf.summary.FileWriter(hparam.log_dir,graph) as logfs:
                # 这里应该有模型恢复操作,现在
                helper.createOrLoadModel(trainModel, hparam)
                sess.run(model.reset_source_op)
                #num_train_steps是外部的总步数,表示要浏览多少次batch,inner_steps是内部步数,
                #浏览一个batch要循环inner_steps次
                inner_steps=hparam.Tmax // hparam.perodic
                for step in range(hparam.num_train_steps):
                    try:
                        #横向扫描输入数据
                        for p in range(inner_steps):
                            global_steps=step*inner_steps+p
                            if (global_steps+5) % hparam.steps_per_state:
                                _, _loss, _acc,_,_lr,_summary=sess.run(stats_op)
                                _update_stat_info(stats_info, _loss, _acc, _lr,global_steps)
                                print_state_info(stats_info)
                                logfs.add_summary(_summary, global_steps)
                                stats_info=_initStatsInfo(hparam)
                            else:
                                _,_loss,_acc,_=sess.run(common_op)
                                _update_stat_info(stats_info,_loss,_acc)
                        #纵向刷新输入数据,并且初始化网络初始状态
                        sess.run([model.feed_source_op,model.transfer_initState_op])
                    except tf.errors.OutOfRangeError:
                        model.saver.save(sess,hparam.model_dir,global_steps)
                        sess.run(model.reset_source_op)


hparam = tf.contrib.training.HParams(
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
    Tmax=6,  # 序列的最大长度,是文件的宽度/特征数量
    lr=1e-4,
    solver='sgd',
    num_train_steps=12000,
    decay_scheme=None,  # "luong5", "luong10", "luong234"
    max_gradient_norm=5,
    features=2,

    src_path='/home/zhangxk/projects/deepAI/ippackage/data/data',
    log_dir='/home/zhangxk/projects/deepAI/ippackage/train/log',
    model_dir='/home/zhangxk/projects/deepAI/ippackage/train/models',
    steps_per_state=10,
    max_keeps=None,
    scope='VPNNetWork',
)
train(hparam)