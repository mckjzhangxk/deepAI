import tensorflow as tf
import model.modelHelper as helper
from model import BaseModel
import time
from utils import print_state_info
import os
import numpy  as np


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

    counter=0
    avg_acc=0
    graph=eval_model.graph
    sess=eval_model.session
    model=eval_model.model

    reflesh_input_op=[model.reset_initState_op,model.feed_source_op]
    acc=[]
    with graph.as_default():
        helper.createOrLoadModel(eval_model,hparam)
        innerstep=hparam.Tmax//hparam.perodic
        sess.run(model.reset_source_op)
        while True:
            try:
                sess.run(reflesh_input_op)
                for s in range(innerstep):
                    __acc,_=sess.run([model.accuracy,model.transfer_initState_op])
                    acc.append(__acc)
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
    
    :param hparam: 
    :return: 
    '''
    trainModel=helper.createTrainModel(hparam,BaseModel)
    evalModel =helper.createEvalModel(hparam, BaseModel)

    model=trainModel.model

    stats_info=_initStatsInfo(hparam)


    common_op=[model.train_op,model.loss,model.accuracy,model.transfer_initState_op]
    stats_op =common_op+ [model.learning_rate,model.summary_op]
    #刷新输入数据, 并且初始化网络初始状态
    refresh_input_op=[model.reset_initState_op,model.feed_source_op]


    with trainModel.session as sess:
        with tf.summary.FileWriter(hparam.log_dir,trainModel.graph) as logfs:

            # 这里应该有模型恢复操作,现在
            helper.createOrLoadModel(trainModel, hparam)
            sess.run(model.reset_source_op)
            #num_train_steps是sess.run的总步数,inner_steps是内部步数,
            #浏览一个batch要循环inner_steps次,outter_step:表示要浏览多少次batch
            #num_train_steps=outter_steps*inner_steps
            inner_steps=hparam.Tmax // hparam.perodic
            outter_steps=hparam.num_train_steps//inner_steps

            for step in range(outter_steps):
                try:
                    #刷新输入数据,并且初始化网络初始状态
                    sess.run(refresh_input_op)

                    #横向扫描输入数据
                    for p in range(inner_steps):
                        global_steps=step*inner_steps+p
                        if (global_steps+5) % hparam.steps_per_state==0:
                            _, _loss, _acc,_,_lr,_summary=sess.run(stats_op)
                            _update_stat_info(stats_info, _loss, _acc, _lr,global_steps)
                            print_state_info(stats_info)
                            logfs.add_summary(_summary, global_steps)
                            stats_info=_initStatsInfo(hparam)
                        else:
                            _,_loss,_acc,_=sess.run(common_op)
                            _update_stat_info(stats_info,_loss,_acc)

                except tf.errors.OutOfRangeError:
                    _save_model(trainModel,hparam.model_dir,global_steps)
                    sess.run(model.reset_source_op)
                    print('eval.................')
                    run_eval(evalModel,hparam,global_steps,logfs)


            print('finish')

            for k in range(3):
                run_eval(evalModel, hparam, global_steps, logfs)
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
    perodic=3,
    Tmax=6,  # 序列的最大长度,是文件的宽度/特征数量
    lr=1e-4,
    solver='adam',
    num_train_steps=500,
    decay_scheme='luong5',  # "luong5", "luong10", "luong234"
    max_gradient_norm=5,
    features=2,

    train_datafile='/home/zhangxk/projects/deepAI/ippackage/data/data1',
    eval_datafile='/home/zhangxk/projects/deepAI/ippackage/data/data1',
    log_dir='/home/zhangxk/projects/deepAI/ippackage/train/log',
    model_dir='/home/zhangxk/projects/deepAI/ippackage/train/models/MyNet',
    steps_per_state=10,
    max_keeps=None,
    scope='VPNNetWork',
    best_model_path=None,

    soft_placement=True,
    log_device=False
)
train(hparam)