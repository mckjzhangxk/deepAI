import tensorflow as tf
import model.modelHelper as helper


class BaseModel():
    def __init__(self,input,hparam):
        self.input=input
        self._setParameter(hparam)

        self._buildNetWork(hparam)
    def _setParameter(self, hparam):
        '''
        把hparam中的参数转化成对象的属性
        :param hparam: 
        :return: 
        '''
        self.mode=hparam.mode
        self.keepProb=hparam.dropout if self.mode=='train' else  0.0
        self.batchSize=hparam.size

    def _buildNetWork(self,hparam):
        pass
    def _buildCellBlock(self,hparam):
        cell = helper.buildRNNCell(
            type=hparam.rnn_type,
            ndims=hparam.ndims,
            number_layers=hparam.num_layers,
            dropout=hparam.dropout,
            forget_bias=hparam.forget_bias,
            residual=hparam.residual
        )
        return cell

hparam=tf.contrib.training.HParams(
    mode='train',
    rnn_type='lstm',
    ndims=128,
    num_layers=2,
    batch=256,
    dropout=0.2,
    forget_bias=1.0,
    residual=False
)
