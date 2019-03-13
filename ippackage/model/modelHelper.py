import tensorflow as tf
from data import get_input
import collections
import os
class RNNState():
    def __cloneState__(self, state, name):
        '''
        state是个tuple,state[i]是个(BATCH,DIM)的tensor,创建和state尺寸相同的variable
        :param state: 
        :return: 
        '''

        clonestate = []
        for i, h in enumerate(state):
            clonestate.append(tf.get_variable('%s_state%d' % (name, i), shape=h.get_shape(), trainable=False,
                                              initializer=tf.zeros_initializer))
        return tuple(clonestate)

    def __init__(self, cell, Batch):
        '''
        根据cell的RNN类型创建初始化状态,每一层是一个对象,保存到self.states里面
        换句话说,self.states的每个状态对于一个层的状态,一个状态是一个tuple包裹,
        针对不同的rnn类型而异

        备注:这个类是维护cell的初始参数,用于解决终了状态传递给初始状态的问题!!

        cell:BasicRNNcell或者MultiRNNCell
        Batch:int,批处理默认长度

        :param cell: 
        :param Batch: 
        '''
        self._states = []
        base_state = cell.zero_state(Batch, tf.float32)

        # 多层RNN
        if isinstance(cell, tf.nn.rnn_cell.MultiRNNCell):
            for n, state in enumerate(base_state):
                name = 'layer%d' % n
                self._states.append(self.__cloneState__(state, name))
        else:  # 单层RNN
            self._states.append(self.__cloneState__(base_state, 'layer0'))

    def get_init_state(self, batch):
        '''
        batch可以是int或者动态的tensor,这里获得初始化的状态,
        返回tuple,tuple[i]可能是具体的tensor或者嵌套tuple
        :param batch: 
        :return: 
        '''
        ret = []
        for state in self._states:
            ret.append(tuple([s[:batch] for s in state]))
        if len(ret) == 1:
            return ret[0]
        return tuple(ret)

    def update_init_state(self, oldState, newState):
        '''
        把oldState的状态更新为newState,oldState[i]如果是tuple,
        要递归调用本方法,更新每一个状态,返回update_op=list(tf.assign)
        :param oldState: 
        :param newState: 
        :return: 
        '''
        update_op = []

        for s1, s2 in zip(oldState, newState):
            if isinstance(s1, tuple):
                sub_update_op = self.update_init_state(s1, s2)
                update_op = update_op + sub_update_op
            else:
                update_op.append(tf.assign(s1, s2))
        return update_op

    def reset_init_state(self, states=None):
        reset_op = []
        if states is None:
            states = self._states

        for state in states:
            if isinstance(state, tuple):
                reset_op = reset_op + self.reset_init_state(state)
            else:
                reset_op.append(tf.assign(state, tf.zeros_like(state)))
        return reset_op
def _redidual_func(inp,out):
    in_dim=inp.get_shape()[-1]
    out_dim=out.get_shape()[-1]

    if in_dim==out_dim:
        return inp+out
    return out

def _singleRNNCell(type, ndims, dropout, forget_bias, residual):
    cell = None
    if type == 'lstm':
        cell = tf.contrib.rnn.BasicLSTMCell(ndims, forget_bias)
    elif type == 'gru':
        cell = tf.contrib.rnn.GRUCell(ndims)
    else:
        raise ValueError('unknown cell type %s' % type)
    if dropout > 0.0:
        cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=1 - dropout)
    if residual:
        cell = tf.contrib.rnn.ResidualWrapper(cell, _redidual_func)
    return cell


def _cells(type, ndims, number_layers, dropout, forget_bias, residual):
    cells = [_singleRNNCell(type, ndims, dropout, forget_bias, residual)
             for n in range(number_layers)]
    return cells


def buildRNNCell(type, ndims, number_layers, dropout, forget_bias, residual):
    '''
    根据给定的type,number_layers,ndims创建一个RNN单元,dropout>0会输入输出都会使用dropout,
    forget_bias是对LSTM的forget_gate的bias初始化,residual表示是否使用resnet把输入和cell输出
    相加.

    返回:
        MultiRNNCell 对象
    :param type: 
    :param ndims: 
    :param number_layers: 
    :param dropout: 
    :param forget_bias: 
    :param residual: 
    :return: 
    '''
    cells = _cells(type, ndims, number_layers, dropout, forget_bias, residual)
    if len(cells) == 1: return cells[0]
    return tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)


def _get_config_proto(hparam):
    conf=tf.ConfigProto(
        allow_soft_placement=hparam.soft_placement,
        log_device_placement=hparam.log_device
    )
    return conf
class VPNModel(collections.namedtuple('VPNModel', ['model', 'graph', 'session'])):pass

def createTrainModel(hparam,ModelFunc):
    return _createModel(hparam,ModelFunc,'run_train')
def createEvalModel(hparam,ModelFunc):
    return _createModel(hparam, ModelFunc, 'eval')
def _createModel(hparam,ModelFunc,mode):
    '''
    根据hparam定义
        输入,模型,sess,graph
    返回:
        VPNModel(model,sess,graph)
    :param hparam: 
    :return: 
    '''
    if mode=='run_train':
        datafile=hparam.train_datafile
    elif mode=='eval':
        datafile=hparam.eval_datafile
    graph = tf.Graph()
    with graph.as_default():
        train_input=get_input(
                  datafile=datafile,
                  BATCH_SIZE=hparam.batch_size,
                  Tmax=hparam.Tmax,
                  D=hparam.features,
                  perodic=hparam.perodic)

        model=ModelFunc(train_input,hparam,mode)
    sess=tf.Session(graph=graph,config=_get_config_proto(hparam))
    return VPNModel(model, graph, sess)


def createOrLoadModel(anymodel,hparam):
    '''
    默认初始化所有variable,如果hparam存在checkpoint,加载参数
    到构造的Graph中.
    
    :param anymodel:这里表示TrainModel,EvalModel,
     包含model,graph,sess 3个属性
    :param hparam: 
    :return: 
    '''
    sess=anymodel.session
    sess.run(tf.global_variables_initializer())

    modeldir=os.path.dirname(hparam.model_dir)
    if tf.train.get_checkpoint_state(modeldir):
        model_path=tf.train.latest_checkpoint(modeldir)
        anymodel.model.saver.restore(sess,model_path)
        print('restore model from path %s'%model_path)