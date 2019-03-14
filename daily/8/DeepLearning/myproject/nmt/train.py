import modelHelper as helper
import tensorflow as tf
from model.BaseModel import Model
import time
import os
from metrics.evaluation_utils import blue
import codecs

hparam=tf.contrib.training.HParams(
    train_src='data/train.vi',
    train_tgt='data/train.en',
    dev_src='data/train.vi',
    dev_tgt='data/train.en',
    test_src='data/train.vi',
    test_tgt='data/train.en',
    vocab_src='data/vocab.vi',
    vocab_tgt='data/vocab.en',

    SOS='<sos>',
    EOS='<eos>',
    UNK='<unk>',
    batch_size=128,

    #####网络相关参数###########
    scope='nmt',
    encode_type='uni',
    rnn_type='lstm',
    emb_size=128,
    ndim=128,
    num_layer=2,
    activation_fn=None,
    dropout=0.0,
    forget_bias=1.0,
    residual_layer=False,
    share_vocab=False,

    infer_mode='beam_search',
    beam_width=3,
    ##########训练参数相关###########
    optimizer='adam',
    lr=1e-2,
    decay_scheme='luong5', #luong5,luong10
    warmup_steps=1000,
    warmup_scheme='t2t',
    max_norm=5,
    ##########训练流程相关###########
    num_train=6000,
    steps_per_stat=10,
    steps_per_innernal_eval=50,
    steps_per_external_eval=100,

    model_path='result/baseModel/model',
    max_to_keep=5,
    ckpt=False,
    checkpoint_path='result/baseModel/model/best',
    log_dir='result/baseModel/log',

    ###########Eval相关参数##############
    subword_option=None,
)
def run_full_eval(eval_model,infer_model,steps,logfs,hparam):
    run_innernal_eval(eval_model,steps,logfs,hparam)
    run_external_eval(infer_model,steps,logfs, hparam)

def run_innernal_eval(evalModel,steps,logfs, hparam):

    '''
    
    使用evalModel对输入计算perplexity,保存最好的模型到hparam.checkpoint_path/perplexity下。
    
    :param evalModel: 
    :param steps: 
    :param logfs: 
    :param hparam: 
    :return: 
    '''
    print('---------------Interval Eval at step %d---------------'%steps)
    model=evalModel.model
    sess=evalModel.session
    keyword = 'perplexity'

    with evalModel.graph.as_default():
        helper.createOrLoadModel(model,sess,hparam)
        perplexity=model.eval(sess)

    _summary=tf.Summary(value=[tf.Summary.Value(tag=keyword,simple_value=perplexity)])
    logfs.add_summary(_summary,steps)

    if not hasattr(hparam,keyword) or getattr(hparam,keyword)>perplexity:
        setattr(hparam,keyword,perplexity)
        model.save(sess,os.path.join(hparam.checkpoint_path,keyword+'.ckpt'))
        print('perplexity:%f,beat the previous!'%perplexity)
    else:
        print('perplexity:%f,not improve the best %f' %(perplexity,getattr(hparam,keyword)))
    print('---------------Finish Interval Eval---------------------')

def run_external_eval(inferModel,steps,logfs, hparam):
    '''
    使用inferModel对输入进行一次翻译,然后根据inferModel.ref_file
    比较，计算blue,保存最好的模型到hparam.checkpoint_path/BLUE_SCORE,
    最好的翻译结果保存到hparam.checkpoint_path/translation.txt下。
    
    :param inferModel: 
    :param steps: 
    :param logfs: 
    :param hparam: 
    :return: 
    '''
    def _output_result(path,translation):
        with codecs.open(path,mode='w',encoding='utf-8') as fs:
            for t in translation:
                fs.write(t+'\n')
    print('---------------External Eval at step %d---------------'%steps)
    keyword='BLUE_SCORE'
    model=inferModel.model
    sess=inferModel.session

    with inferModel.graph.as_default():
        helper.createOrLoadModel(model,sess,hparam)
        rs=model.infer(sess)
    blue_score=blue(rs.translation,inferModel.ref_file,hparam.subword_option)

    _summary=tf.Summary(value=[tf.Summary.Value(tag=keyword,simple_value=blue_score)])
    logfs.add_summary(_summary,steps)

    if not hasattr(hparam,keyword) or getattr(hparam,keyword)<blue_score:
        setattr(hparam,keyword,blue_score)
        model.save(sess,os.path.join(hparam.checkpoint_path,keyword+'.ckpt'))
        _output_result(os.path.join(hparam.checkpoint_path,'translation.txt'),rs.translation)
        print('BLUE SCORE:%f,beat the previous!'%blue_score)
    else:
        print('BLUE SCORE:%f,not improve the best %f' %(blue_score,getattr(hparam,keyword)))

    print('---------------Finish External Eval---------------------')

def __init_stat_info__(hparam):
    start_time=time.time()

    return {
        'start_time':start_time,
        'loss':0,
        'global_norm':0,
        'words':0,
        'lr':0
    }
def __updata_stat_info__(stat,trainoutput):
    stat['loss']+=trainoutput.loss
    stat['global_norm']+=trainoutput.global_norm
    stat['words']+=trainoutput.word_count
    stat['lr']=trainoutput.lr

def __print_stat_info__(stat,globalstep,hparam):
    steps=hparam.steps_per_stat
    loss=stat['loss']/steps
    norm=stat['global_norm']/steps
    lr=stat['lr']
    speed=stat['words']/(time.time()-stat['start_time'])
    s='step %d,avg loss:%.f,avg_grad_norm:%.3f,lr:%f,speed:%d wps'%(globalstep,loss,norm,lr,speed)
    print(s)
def _chooseModel(hparam):
    return Model
def train(hparam):
    modelFunc=_chooseModel(hparam)

    train_model=helper.createTrainModel(hparam,modelFunc)
    eval_model=helper.createEvalModel(hparam,modelFunc)
    infer_model=helper.createInferModel(hparam,modelFunc,hparam.dev_src)

    stat_info=__init_stat_info__(hparam)


    with train_model.session as sess:
        with tf.summary.FileWriter(hparam.log_dir) as fs:
            helper.createOrLoadModel(train_model.model, sess, hparam)
            sess.run(train_model.batch_input.initializer)
            for step in range(hparam.num_train):
                try:
                    rs=train_model.model.train(sess)
                    __updata_stat_info__(stat_info,rs)
                    if rs.global_step% hparam.steps_per_stat==0:
                        fs.add_summary(rs.summary,rs.global_step)
                        __print_stat_info__(stat_info,rs.global_step,hparam)
                        stat_info=__init_stat_info__(hparam)

                    if rs.global_step% hparam.steps_per_innernal_eval==0:
                        run_innernal_eval(eval_model,rs.global_step,fs,hparam)
                    if rs.global_step % hparam.steps_per_external_eval == 0:
                        run_external_eval(infer_model,rs.global_step,fs,hparam)
                except tf.errors.OutOfRangeError:
                    sess.run(train_model.batch_input.initializer)
                    train_model.model.save(sess,hparam.model_path,rs.global_step)
                    # run_full_eval(eval_model,infer_model,rs.global_step,fs,hparam)
train(hparam)