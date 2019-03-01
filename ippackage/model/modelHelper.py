import tensorflow as tf
from data import get_input
import collections

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
        cell = tf.contrib.rnn.ResidualWrapper(cell, None)
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


def accuracy(logit,label,valid_idx=None):
    '''
    logit:是一个(batch,C)的tensor
    label:(batch,)标注
    valid:(batch,)如果是None,默认都是有效的
    
    返回准确率
    :param logit: 
    :param label: 
    :param valid_idx: 
    :return: 
    '''

    if valid_idx is None:
        valid_idx=tf.to_float(tf.ones_like(logit))
    else:
        valid_idx=tf.to_float(valid_idx)

    predict=tf.arg_max(logit,1,output_type=tf.int32)
    true_predict=tf.to_float(tf.equal(predict,label))
    true_predict_cnt=tf.reduce_sum(true_predict)
    cnt=tf.reduce_sum(valid_idx)+1e-13
    return tf.divide(true_predict_cnt,cnt)

def _get_config_proto():
    conf=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True
    )
    return conf
class TrainModel(collections.namedtuple('TrainModel',['model','graph','session'])):pass
def createTrainModel(hparam,Model):
    '''
    根据hparam定义
        输入,模型,sess,graph
    返回:
        TrainModel(model,sess,graph)
    :param hparam: 
    :return: 
    '''
    assert hparam.mode=='train','You must in train Model'
    train_input=get_input(datafile=hparam.src_path,
              BATCH_SIZE=hparam.batch_size,
              Tmax=hparam.Tmax,
              D=hparam.features,
              perodic=hparam.perodic)

    model=Model(train_input,hparam)

    graph=tf.Graph()

    sess=tf.Session(graph=graph,config=_get_config_proto())

    return TrainModel(model,graph,sess)


def createOrLoadModel(anymodel,model,hparam):
    sess=anymodel.session
    sess.run(tf.global_variables_initializer())

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

    src_path='/home/zhangxk/projects/deepAI/ippackage/data/data'
)