import tensorflow as tf
from utils.iterator_utils import get_iterator,get_infer_iterator
from utils.vocab_utils import check_vocab
import collections
import os

def _residual_func(inp,out):
    return inp+out

def _sigle_rnn_cell(hparam,dropout):
    '''
    hparam:
        rnn_type:lstm|gru
        ndim:default 128
        dropout:
        residual
        activation:
    返回一个cell
    :param hparam: 
    :return: 
    '''
    rnn_type=hparam.rnn_type.lower()
    ndim=hparam.ndim
    residual=hparam.residual_layer
    activation= hparam.activation_fn

    if rnn_type=='lstm':
        _cell=tf.contrib.rnn.BasicLSTMCell(ndim,
                                     activation=activation,
                                     forget_bias=hparam.forget_bias,
                                     state_is_tuple=True)

    elif rnn_type=='gru':
        _cell=tf.contrib.rnn.GRUCell(ndim)
    else:
        raise ValueError('Unkwon Rnn Cell Type %s'%rnn_type)

    if dropout>0.0:
        _cell=tf.contrib.rnn.DropoutWrapper(_cell,input_keep_prob=1-dropout)
    if residual:
        _cell=tf.contrib.rnn.ResidualWrapper(_cell,_residual_func)
    return _cell


def _cell_list(hparam,dropout):
    '''
    定义了RNN 的单元 返回
    list of rnn_cell
    :param hparam: 
    :return: 
    '''
    num_layer=hparam.num_layer

    cells=[_sigle_rnn_cell(hparam,dropout) for n in range(num_layer)]

    return cells
def create_rnn_cell(hparam,dropout):
    '''
        hparam:
        num_layer:
        rnn_type:lstm|gru
        ndim:default 128
        dropout:
        residual
        activation:
    
    :param hparam: 
    :return: 
    '''
    _cells=_cell_list(hparam,dropout)
    if len(_cells)==1:
        return _cells[0]
    else:
        return tf.contrib.rnn.MultiRNNCell(_cells)

def create_emb_matric(hparam):
    '''
    
    :param hparam: 
    :return: 
    '''

    src_size,_=check_vocab(hparam.vocab_src)
    tgt_size,_=check_vocab(hparam.vocab_tgt)
    emb_size = hparam.emb_size
    share_vocab=hparam.share_vocab

    if share_vocab:
        if src_size!=tgt_size:
            raise ValueError('can not share vocab,because src.Vsize !=tgt.Vsize')
        emb_matric=tf.get_variable('embeding',shape=(src_size,emb_size))
        return (emb_matric,emb_matric)
    else:
        encode_matric=tf.get_variable('embeding/encoder',shape=(src_size,emb_size))
        decoder_matric=tf.get_variable('embeding/decoder',shape=(tgt_size,emb_size))
        return (encode_matric,decoder_matric)

class TrainModel(collections.namedtuple('TrainModel',['model','session','graph','batch_input'])):pass
class EvalModel(collections.namedtuple('EvalModel',['model','session','graph','batch_input'])):pass
class InferModel(collections.namedtuple('InferModel',['model','session','graph','batch_input','ref_file'])):pass
# class TrainModel(collections.namedtuple('TrainModel',['model','session','graph','batch_input'])):pass



def createOrLoadModel(model,sess,hparam):
    '''
    如果hparam.checkpoint_path设置了，那么从checkpoint_path
    回复参数
    否在从hparam.model_path
    如果hparam.model_path不包含有效参数,run global_variables_initializer
    :param model: 
    :param sess: 
    :param hparam: 
    :return: 
    '''

    model_path=None
    if hparam.ckpt:
        model_path=hparam.checkpoint_path
    elif tf.train.get_checkpoint_state(os.path.dirname(hparam.model_path)):
        model_path=tf.train.latest_checkpoint(os.path.dirname(hparam.model_path))

    if model_path is None:
        sess.run(tf.global_variables_initializer())
    else:
        model.restore(sess,model_path)
        print('recover model from %s'%model_path)
    sess.run(tf.tables_initializer())


def _createModel(mode, hparam, modelFunc=None):
    '''
    根据hparam:
        train_src,train_tgt,创建数据集，然会返回TrainModel
        TrainModels是和神经网络输入相关的 封装
    :param hparam: 
    :return: 
    '''
    def _get_config_proto():
        conf=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
        )
        return conf

    graph=tf.Graph()
    with graph.as_default():

        if mode=='train':
            src_dataset=tf.data.TextLineDataset(hparam.train_src)
            tgt_dataset=tf.data.TextLineDataset(hparam.train_tgt)
        elif mode=='eval':
            src_dataset=tf.data.TextLineDataset(hparam.dev_src)
            tgt_dataset=tf.data.TextLineDataset(hparam.dev_tgt)
        else:
            raise ValueError('_createTrainModel.mode must be train or eval')
        batch_input=get_iterator(src_dataset,tgt_dataset,hparam)
        sess=tf.Session(config=_get_config_proto())
        model=modelFunc(batch_input,mode,hparam)


        if mode=='train':
            return TrainModel(model,sess,graph,batch_input)
        elif mode=='eval':
            return EvalModel(model, sess, graph, batch_input)


def createTrainModel(hparam,modelFunc=None):
    '''
    根据hparam:
        train_src,train_tgt,创建数据集，然会返回TrainModel
        TrainModels是和神经网络输入相关的 封装
            model,graph,sess,batch_input
    :param hparam: 
    :return: 
    '''
    return _createModel('train', hparam, modelFunc)

def createEvalModel(hparam,modelFunc=None):
    return _createModel('eval', hparam, modelFunc)

def createInferModel(hparam,modelFunc=None,src_path=None,tgt_path=None):
    def _get_config_proto():
        conf=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
        )
        return conf

    graph=tf.Graph()
    with graph.as_default():
        if src_path:
            src_dataset = tf.data.TextLineDataset(src_path)
        else:
            src_dataset=tf.data.TextLineDataset(hparam.test_src)
        if tgt_path:
            ref_file = tgt_path
        else:
            ref_file=hparam.test_tgt

        batch_input=get_infer_iterator(src_dataset,hparam)
        sess=tf.Session(config=_get_config_proto())
        model=modelFunc(batch_input,'infer',hparam)

        return InferModel(model,sess,graph,batch_input,ref_file)
