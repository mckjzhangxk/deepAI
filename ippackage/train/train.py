import tensorflow as tf
import model.modelHelper as helper
from model import BaseModel
import time
from utils import print_state_info
import os
import numpy as np


def _initStatsInfo(hparam):
    start_time=time.time()

    return {
        'start_time':start_time,
        'total_loss':[],
        'accuracy':[],
        'learn_rate':0.0,
        'stat_steps':hparam.steps_per_state
    }
def _update_stat_info(stat_info,loss,accuracy,lr=None,globalstep=None):
    stat_info['total_loss'].append(loss)
    stat_info['accuracy'].append(accuracy)

    if globalstep is not None:
        stat_info['steps']=globalstep
    if lr:
        stat_info['learn_rate']=lr

def _save_model(anymodel,path,steps=None):

    anymodel.model.saver.save(anymodel.session,path,global_step=steps)

def run_eval(eval_model,hparam,globalstep,writer):
    '''
    使用eval_model,对测试集的数据运行一边计算accuracy,
    打印到控制台,并用writer记录到日志中.如果模型accuracy达到最好的时候,保存模型
    到hparam.best_path下面的best_acc文件
    :param eval_model: 
    :param hparam: 
    :return: 
    '''

    graph=eval_model.graph
    sess=eval_model.session
    model=eval_model.model

    acc=[]
    with graph.as_default():
        helper.createOrLoadModel(eval_model,hparam)
        model.init_dataSource(sess)
        while True:
            try:
                result=model.eval(sess,hparam)
                acc.append(result['accuracy'])
            except tf.errors.OutOfRangeError:
                break
    #对所有batch的统计结果
    avg_acc=np.average(acc)

    if getattr(hparam,'best_accuracy',0.0)<avg_acc:
        setattr(hparam,'best_accuracy',avg_acc)
        best_model_path=getattr(hparam,'best_model_path',None)
        if best_model_path:
            _save_model(eval_model,
                        os.path.join(best_model_path,'best_acc'))
    #结果打印到控制台和日志中,便于tensorboard分析查看
    _summary=tf.Summary(value=[tf.Summary.Value(tag='EvalAcc',simple_value=avg_acc)])
    writer.add_summary(_summary,globalstep)
    print('After step %d,Evaluation Accuracy is %f'%(globalstep,avg_acc))

def train(hparam):
    '''
    创建 train,eval两个计算图,读取数据集进行训练,数据集结束的
    时候eval,统计准确性.
    每steps_per_state打印在这个区间内计算的平均loss,accuarcy
    
    :param hparam: 
    :return: 
    '''
    trainModel=helper.createTrainModel(hparam,BaseModel)
    evalModel =helper.createEvalModel(hparam, BaseModel)

    model=trainModel.model

    stats_info=_initStatsInfo(hparam)


    with trainModel.session as sess:
        with tf.summary.FileWriter(hparam.log_dir,trainModel.graph) as logfs:

            # 这里应该有模型恢复操作,现在
            helper.createOrLoadModel(trainModel, hparam)

            #num_train_steps是sess.run的总步数,inner_steps是内部步数,
            #浏览一个batch要循环inner_steps次,outter_step:表示要浏览多少次batch
            #num_train_steps=outter_steps*inner_steps
            inner_steps=hparam.Tmax // hparam.perodic
            outter_steps=hparam.num_train_steps//inner_steps

            model.init_dataSource(sess)
            for step in range(outter_steps):
                try:
                    result=model.train(sess,hparam)
                    _loss, _acc,_globalstep,_lr,_summary=result['result']
                    _update_stat_info(stats_info, _loss, _acc,_lr)

                    if result['stat']:
                        print_state_info(stats_info)
                        logfs.add_summary(_summary, _globalstep)
                        stats_info = _initStatsInfo(hparam)


                except tf.errors.OutOfRangeError:
                    _save_model(trainModel,hparam.model_dir,_globalstep)
                    model.init_dataSource(sess)

                    print('eval.................')
                    run_eval(evalModel,hparam,_globalstep,logfs)

hparam = tf.contrib.training.HParams(
    mode='train',
    rnn_type='lstm',
    ndims=128,
    num_layers=2,
    num_output=2,
    batch_size=128,
    dropout=0.0,
    forget_bias=1.0,
    residual=False,
    perodic=30,
    Tmax=300,  # 序列的最大长度,是文件的宽度/特征数量
    lr=1e-4,
    solver='adam',
    num_train_steps=5000,
    decay_scheme='luong5',  # "luong5", "luong10", "luong234"
    max_gradient_norm=5,
    features=6,

    train_datafile='/home/zhangxk/AIProject/ippack/vpndata/train.txt',
    eval_datafile='/home/zhangxk/AIProject/ippack/vpndata/train.txt',
    log_dir='/home/zhangxk/projects/deepAI/ippackage/train/log',
    model_dir='/home/zhangxk/projects/deepAI/ippackage/train/models/MyNet',
    steps_per_state=10,
    max_keeps=5,
    scope='VPNNetWork',
    best_model_path='/home/zhangxk/projects/deepAI/ippackage/train/best',

    soft_placement=True,
    log_device=False
)
train(hparam)