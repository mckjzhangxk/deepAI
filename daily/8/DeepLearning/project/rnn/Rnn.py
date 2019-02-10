import tensorflow as tf


class RNN():

    def __init__(self,T=40,input_dim=1000,hidden_dim=512,name='rnn'):
        '''
        :param T:how many steps
        :param input_dim:
        :param hidden_dim:
        :param name: variable scope name
        :return NOne
        :define Wa,Wx,Ba,with shape[hidde,hidden],[input,hidden],[hidde]
        '''
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):

            self.wa=tf.get_variable('Wa',shape=[hidden_dim,hidden_dim])
            self.ba=tf.get_variable('ba',shape=[hidden_dim],initializer=tf.zeros_initializer())
            self.wx=tf.get_variable('Wx',shape=[input_dim,hidden_dim])
            self.T=T
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim

    def _rnnCell(self,X,a0,f=None):
        '''

        :param X:shape[None,input_dims]
        :param a0:previous status,shape[None,hidden_dims]
        :return:
        '''

        assert X.shape[1]==self.input_dim,'asset input must have dim %d'%self.input_dim

        a1=tf.matmul(X,self.wa)+tf.matmul(a0,self.wx)+self.ba
        if f:
            a1=f(a1)
        return a1

    def rnn(self,X,a0,f=tf.nn.tanh):
        '''

        :param X:shape[None,T,inputdim]
        :param a0:shape[None,hidden_dim]
        :param f: none linear function
        :return:
        '''
        hidden_states=[]
        for t in range(self.T):
            a0=self._rnnCell(X[:,t,:],a0,f)
            hidden_states.append(a0)
        return hidden_states