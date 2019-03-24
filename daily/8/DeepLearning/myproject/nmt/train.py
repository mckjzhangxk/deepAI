
import modelHelper as helper
import tensorflow as tf
import time
import os
from metrics.evaluation_utils import blue
import codecs

hparam=tf.contrib.training.HParams(
    train_src='/home/zxk/AI/nmt/nmt/nmt_data/train.vi',
    train_tgt='/home/zxk/AI/nmt/nmt/nmt_data/train.en',
    dev_src='/home/zxk/AI/nmt/nmt/nmt_data/tst2012.vi',
    dev_tgt='/home/zxk/AI/nmt/nmt/nmt_data/tst2012.en',
    test_src='/home/zxk/AI/nmt/nmt/nmt_data/st2013.vi',
    test_tgt='/home/zxk/AI/nmt/nmt/nmt_data/st2013.en',
    vocab_src='/home/zxk/AI/nmt/nmt/nmt_data/vocab.vi',
    vocab_tgt='/home/zxk/AI/nmt/nmt/nmt_data/vocab.en',
    # train_src='data/train.vi',
    # train_tgt='data/train.en',
    # dev_src='data/train.vi',
    # dev_tgt='data/train.en',
    # test_src='data/train.vi',
    # test_tgt='data/train.en',
    # vocab_src='data/vocab.vi',
    # vocab_tgt='data/vocab.en',

    src_max_len=50,
    tgt_max_len=50,
    SOS='<s>',
    EOS='</s>',
    UNK='<unk>',
    batch_size=128,

    #####网络模型相关参数###########
    scope='nmt',
    dtype=tf.float32,
    encode_type='uni',
    rnn_type='lstm',
    layer_norm=False,
    emb_size=512,
    ndim=512,
    num_layer=2,
    activation_fn=tf.nn.tanh,
    dropout=0.2,
    forget_bias=1.0,
    residual_layer=False,
    share_vocab=False,
    nsample_softmax=2000,

    infer_mode='greedy',
    beam_width=1,

    #atten_type='luong',
    pass_hidden_state=True,
    ##########训练参数相关###########
    optimizer='sgd',
    lr=1.0,
    decay_scheme='luong10', #luong5,luong10
    warmup_steps=100,
    warmup_scheme='t2t',
    max_norm=5,
    ##########训练流程相关###########
    num_train=12000,
    steps_per_stat=10,
    steps_per_innernal_eval=200,
    steps_per_external_eval=1000,

    model_path='result/baseModel/model',
    max_to_keep=5,
    ckpt=False,
    checkpoint_path='result/baseModel/best',
    log_dir='result/baseModel/log',
    avg_ckpt=True,
    ###########Eval相关参数##############
    #subword_option='bpe',
    subword_option=None,
    BLUE_SCORE=0.0,
    perplexity=10000000.0

)
def cal_param_cnt(model):
    import numpy as np
    cnt=0
    with model.graph.as_default():
        for v in tf.trainable_variables():
            cnt+=np.prod(v.shape.as_list())
    print('total have %d paramaters'%cnt)

def run_full_eval(eval_model,infer_model,steps,logfs,hparam):
    run_innernal_eval(eval_model,steps,logfs,hparam)
    run_external_eval(infer_model,steps,logfs, hparam)
def _innernal_eval(evalModel,steps,logfs, hparam):
    model = evalModel.model
    sess = evalModel.session
    keyword = 'perplexity'

    with evalModel.graph.as_default():
        perplexity = model.eval(sess)
    _summary = tf.Summary(value=[tf.Summary.Value(tag=keyword, simple_value=perplexity)])
    logfs.add_summary(_summary, steps)

    if not hasattr(hparam, keyword) or getattr(hparam, keyword) > perplexity:
        setattr(hparam, keyword, perplexity)
        print('perplexity:%f,beat the previous!' % perplexity)
        model.save(sess, os.path.join(hparam.checkpoint_path, keyword + '.ckpt'))
    else:
        print('perplexity:%f,not improve the best %f' % (perplexity, getattr(hparam, keyword)))

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
    graph=evalModel.graph


    helper.createOrLoadModel(model,graph,sess,hparam)
    _innernal_eval(evalModel, steps, logfs, hparam)
    if hparam.avg_ckpt:
        print('run avg internal eval....')
        helper.avg_Ckpt_Of_Model(graph,sess,hparam)
        _innernal_eval(evalModel,steps,logfs,hparam)

    print('---------------Finish Interval Eval---------------------')

def _external_eval(inferModel,steps,logfs, hparam):
    def _output_result(path,translation):
        with codecs.open(path,mode='w',encoding='utf-8') as fs:
            for t in translation:
                fs.write(t+'\n')
    keyword='BLUE_SCORE'
    model=inferModel.model
    sess=inferModel.session
    graph=inferModel.graph

    with graph.as_default():
        rs=model.infer(sess)
    blue_score=blue(rs.translation,inferModel.ref_file,hparam.subword_option)
    if logfs:
        _summary=tf.Summary(value=[tf.Summary.Value(tag=keyword,simple_value=blue_score)])
        logfs.add_summary(_summary,steps)

    if not hasattr(hparam,keyword) or getattr(hparam,keyword)<blue_score:
        setattr(hparam,keyword,blue_score)
        _output_result(os.path.join(hparam.checkpoint_path,'translation.txt'),rs.translation)
        print('BLUE SCORE:%f,beat the previous!'%blue_score)
        model.save(sess,os.path.join(hparam.checkpoint_path,keyword+'.ckpt'))

    else:
        print('BLUE SCORE:%f,not improve the best %f' %(blue_score,getattr(hparam,keyword)))

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

    print('---------------External Eval at step %d---------------'%steps)
    model=inferModel.model
    sess=inferModel.session
    graph=inferModel.graph

    helper.createOrLoadModel(model,graph,sess,hparam)
    _external_eval(inferModel, steps, logfs, hparam)

    if hparam.avg_ckpt:
        print('run avg external eval....')
        helper.avg_Ckpt_Of_Model(graph,sess,hparam)
        _external_eval(inferModel, steps, logfs, hparam)
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

def train(hparam):
    modelFunc=helper.chooseModel(hparam)

    train_model=helper.createTrainModel(hparam,modelFunc)
    cal_param_cnt(train_model)
    eval_model=helper.createEvalModel(hparam,modelFunc)
    infer_model=helper.createInferModel(hparam,modelFunc,hparam.dev_src,hparam.dev_tgt)
    with train_model.graph.as_default():
        train_model.session.run(tf.tables_initializer())
    with eval_model.graph.as_default():
        eval_model.session.run(tf.tables_initializer())
    with infer_model.graph.as_default():
        infer_model.session.run(tf.tables_initializer())
    stat_info=__init_stat_info__(hparam)


    with train_model.session as sess:
        with tf.summary.FileWriter(hparam.log_dir) as fs:
            helper.createOrLoadModel(train_model.model,train_model.graph, sess, hparam)
            train_model.model.do_Initialization(sess, hparam)
            for step in range(hparam.num_train):
                try:
                    rs=train_model.model.train(sess)
                    __updata_stat_info__(stat_info,rs)
                    if rs.global_step% hparam.steps_per_stat==0:
                        fs.add_summary(rs.summary,rs.global_step)
                        __print_stat_info__(stat_info,rs.global_step,hparam)
                        stat_info=__init_stat_info__(hparam)

                    if rs.global_step% hparam.steps_per_innernal_eval==0:
                        train_model.model.save(sess,hparam.model_path,rs.global_step)
                        run_innernal_eval(eval_model,rs.global_step,fs,hparam)
                    if rs.global_step % hparam.steps_per_external_eval == 0:
                        run_external_eval(infer_model,rs.global_step,fs,hparam)
                except tf.errors.OutOfRangeError:
                    train_model.model.do_Initialization(sess,hparam)
                    train_model.model.save(sess,hparam.model_path,rs.global_step)
                    run_full_eval(eval_model,infer_model,rs.global_step,fs,hparam)

    #final report test result
    #infer_model = helper.createInferModel(hparam, modelFunc, hparam.dev_src)
    #_external_eval(infer_model,0,None,hparam)
train(hparam)
