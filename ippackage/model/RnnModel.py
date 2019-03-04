import tensorflow as tf
import model.modelHelper as helper
from metrics import accuracy,accuracyPerClass

class BaseModel():
    def __init__(self,input,hparam,mode):
        self._input=input
        self._mode=mode
        self._setParameter(hparam)
        self._buildNetWork(hparam)
        if self._mode=='train':
            self._set_train(hparam)
            self._summary()
        elif self._mode=='eval':
            self._set_eval(hparam)

        self._setSaver(hparam)
    def _setParameter(self, hparam):
        '''
        把hparam中的参数转化成对象的属性
        :param hparam: 
        :return: 
        '''
        self._dropout=hparam.dropout if self._mode == 'train' else  0.0
    def _setSaver(self,hparam):
        varlist=[]
        for v in tf.global_variables():
            if v.name.startswith(hparam.scope):
                varlist.append(v)
        self.saver=tf.train.Saver(varlist,max_to_keep=hparam.max_keeps)
    def _buildNetWork(self,hparam):
        '''
        常见基本网络,流程:
        1.创建RNN CELL
        2.初始化网络的初始状态对象,以及重置初始化操作
        3.根据网络长度,创建网络
        4.创建初始状态传递操作
        5.映射rnn_state到输出
        
        以上过程产生的对象:
            2.) _initstate:rnn的初始状态,tuple(Variable(BATCH,ndims,float32)....)
               _reset_init_state_op:执行sess.run后initstate被重置
            3.)T(int)
               _states(?,T,ndims,float32)
            4.)_transfer_state_op:执行sess.run后initstate被赋值
            5.)_logit(?,num_output,float32)
        :param hparam: 
        :return: 
        '''
        def _getRNNLength_():
            assert hparam.Tmax%hparam.perodic==0,'Tmax must be a multiper of perodic'
            assert hparam.Tmax>=hparam.perodic,'Tmax must greater than perodic'
            return hparam.perodic

        with tf.variable_scope(hparam.scope):
            # 获得rnn_cell
            cell=self._buildCellBlock(hparam)

            #创建一个初始状态的维护对象,注意下面2个batch的区别,一个是静态,一个是动态
            rnnstateObj=helper.RNNState(cell,hparam.batch_size)
            init_state=rnnstateObj.get_init_state(self._input.batch_size)
            self._initstate = init_state
            self._reset_init_state_op = rnnstateObj.reset_init_state()

            # 获得的RNN序列长度,然后创建序列网络
            self.T = _getRNNLength_()
            _states=[]
            for t in range(self.T):
                _,init_state=cell(self._input.X[:,t],init_state)
                _states.append(init_state[-1][-1])
            #self._states.shape=(self.input.batch_size,self.T,ndims)
            self._states=tf.stack(_states,axis=1)

            #这里的init_state可以看作是final state
            self._transfer_state_op=rnnstateObj.update_init_state(self._initstate, init_state)



            #映射函数
            proj=tf.layers.Dense(hparam.num_output,
                            activation=None,
                            use_bias=True)
            _logit=proj(self._states)
            _cursor=self._input.Cursor
            #计算取logit的位置
            #A)如果输入没有在本段RNN结束,那么应该取self.T-1的状态取为logit
            #B)如果输入恰好在本段RNN结束,那么seqlen-cursor-1的状态应该取为logit
            #C)如果本段RNN没有的输入已经结束,那么不应该有logit,但是为了满足计算,默认取0
                # 设置valid标志,计算loss的时候过滤掉这一loss
            #D)计算有效的idx一定要依赖_logit,因为计算_logit才会刷新输入数据的指针
            with tf.control_dependencies([_logit]):
                _logit_idx=tf.minimum(self._input.X_len-_cursor-1,self.T-1)
                self.xxx=_logit_idx
                _valid_idx=tf.cast(tf.greater_equal(_logit_idx,0),tf.float32)
                _logit_idx=tf.maximum(_logit_idx,0)
                self._valid_idx=_valid_idx


                #logit(self.input.batch_size,self.T,num_output,float32)
                i1=tf.range(self._input.batch_size)
                i2=_logit_idx
                ii=tf.stack([i1,i2],axis=1)
                self._logit=tf.gather_nd(params=_logit,indices=ii)

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


    def _get_learning_rate_warmup(self, hparams):
        """Get learning rate warmup."""
        warmup_steps = hparams.warmup_steps
        warmup_scheme = hparams.warmup_scheme


        # Apply inverse decay if global steps less than warmup steps.
        # Inspired by https://arxiv.org/pdf/1706.03762.pdf (Section 5.3)
        # When step < warmup_steps,
        #   learing_rate *= warmup_factor ** (warmup_steps - step)
        if warmup_scheme == "t2t":
            # 0.01^(1/warmup_steps): we start with a lr, 100 times smaller
            warmup_factor = tf.exp(tf.log(0.01) / warmup_steps)
            inv_decay = warmup_factor ** (
                tf.to_float(warmup_steps - self.global_step))
        else:
            raise ValueError("Unknown warmup scheme %s" % warmup_scheme)

        return tf.cond(
            self.global_step < hparams.warmup_steps,
            lambda: inv_decay * self._learning_rate,
            lambda: self._learning_rate,
            name="learning_rate_warump_cond")

    def _get_decay_info(self, hparams):
        '''
        根据hparam.decay_scheme,返回开始
            start_decay_step:开始从哪里decay
            decay_steps:隔多少步decay
            decayFactor(0.5或者1.0)
            
        :param hparams: 
        :return: 
        '''
        """Return decay info based on decay_scheme."""
        if hparams.decay_scheme in ["luong5", "luong10", "luong234"]:
            decay_factor = 0.5
            if hparams.decay_scheme == "luong5":
                start_decay_step = int(hparams.num_train_steps / 2)
                decay_times = 5
            elif hparams.decay_scheme == "luong10":
                start_decay_step = int(hparams.num_train_steps / 2)
                decay_times = 10
            elif hparams.decay_scheme == "luong234":
                start_decay_step = int(hparams.num_train_steps * 2 / 3)
                decay_times = 4
            remain_steps = hparams.num_train_steps - start_decay_step
            decay_steps = int(remain_steps / decay_times)
        elif not hparams.decay_scheme:  # no decay
            start_decay_step = hparams.num_train_steps
            decay_steps = 0
            decay_factor = 1.0
        elif hparams.decay_scheme:
            raise ValueError("Unknown decay scheme %s" % hparams.decay_scheme)
        return start_decay_step, decay_steps, decay_factor

    def _get_learning_rate_decay(self, hparams):
        '''
            返回decay学习率
                global_steps<start_decay_step:lr 不变
                start_decay_step>global_steps:
                lr=lr*decay_factor**(global_steps-start_decay_step//decay_steps)

                从start_decay_steps后,每隔decay_steps步,lr=lr*decay_factor,
                decay_factor=0.5,默认不衰减

                luong234:训练过2/3后,减半4次
                luong5:训练过1/2后,减半5次
                luong10:训练过1/2后,减半10次
        :param hparams: 
        :return: 
        '''
        """Get learning rate decay."""
        start_decay_step, decay_steps, decay_factor = self._get_decay_info(hparams)
        return tf.cond(
            self.global_step < start_decay_step,
            lambda: self._learning_rate,
            lambda: tf.train.exponential_decay(
                self._learning_rate,
                (self.global_step - start_decay_step),
                decay_steps, decay_factor, staircase=True),
            name="learning_rate_decay_cond")

    def _set_train(self,hparam):
        '''
        设置训练过程,分为如下步骤:
        1.loss目标函数建立
        2.学习率的设在,这里可以使用warup策略(hparam.warmup_scheme)
                    以及学习率递减策略(hparam.decay_scheme)
        3.创建优化器hparam.solver
        4.优化细则:对于梯度进行clip,如果全局grad_norm>hparam.max_grad_norm,
            grad缩放max_grad_norm/global_grad_norm
        5.设在优化步骤train_op
        
        以上过程产生如下全局变量:
            1.)self._loss
            2.)self._learning_rate,self.global_step
            4.)self._clip_grad:所有的梯度
               self._global_norm:全局梯度大小(未clip)
            5.)self.train_op
        :param hparam: 
        :return: 
        '''
        def __getSolver__(name,lr):
            if name=='sgd':
                return tf.train.GradientDescentOptimizer(lr)
            elif name=='adam':
                return tf.train.AdamOptimizer(lr)
            elif name=='rmsp':
                return tf.train.RMSPropOptimizer(lr)
            else:
                raise ValueError('unknown solver')

        _loss=tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self._input.Y,
            logits=self._logit
        )*self._valid_idx
        self._loss=tf.reduce_mean(_loss)
        self._accuracy=accuracy(logit=self._logit,
                                label=self._input.Y,
                                valid_idx=self._valid_idx)
        self._clsAccuracy=accuracyPerClass(logit=self._logit,
                                           label=self._input.Y,
                                           valid_idx=self._valid_idx,
                                           C=hparam.num_output
                                           )
        #2.学习率与学习策略
        self.global_step=tf.Variable(0,trainable=False,name='global_step')
        self._learning_rate=tf.constant(hparam.lr)
        if hasattr(hparam,'warmup_scheme'):
            self._learning_rate=self._get_learning_rate_warmup(hparam)
        self._learning_rate=self._get_learning_rate_decay(hparam)

        #3.优化器
        optimizer=__getSolver__(hparam.solver, self._learning_rate)

        #4.梯度处理,防止梯度explode
        varlist=tf.trainable_variables()
        _gradient=tf.gradients(self._loss,varlist)
        self._clip_grad,self._global_gradient_norm=\
            tf.clip_by_global_norm(_gradient,hparam.max_gradient_norm)

        #5.优化操作
        self._train_op=optimizer.apply_gradients(zip(self._clip_grad,varlist),self.global_step)

    def _set_eval(self,hparam):
        '''
        
        :param hparam: 
        :return: 
        '''
        self._accuracy=accuracy(logit=self._logit,
                                label=self._input.Y,
                                valid_idx=self._valid_idx)
        self._clsAccuracy=accuracyPerClass(logit=self._logit,
                                           label=self._input.Y,
                                           valid_idx=self._valid_idx,
                                           C=hparam.num_output
                                           )
        tf.summary.scalar('eval_accuracy', self._accuracy)
        for c,v in enumerate(self._clsAccuracy):
            tf.summary.scalar('eval_c%d_acc'%c,v)

    def _summary(self):
        '''
        对learning_rate,loss,global_norm,clip_norm做总结
        :return: 
        '''

        tf.summary.scalar('learn_rate',self._learning_rate)
        tf.summary.scalar('loss',self._loss)
        tf.summary.scalar('global_grad_norm', self._global_gradient_norm)
        tf.summary.scalar('accuracy', self._accuracy)
        for v in self._clsAccuracy:tf.summary.scalar(v.name,v)
        for v in self._clip_grad:tf.summary.histogram(v.name,v)
        tf.summary.scalar('clip_grad_norm',tf.global_norm(self._clip_grad))

        self._summary_op=tf.summary.merge_all()

    @property
    def feed_source_op(self):
        return self._input.Update_Source
    @property
    def reset_source_op(self):
        return self._input.Iterator.initializer
    @property
    def summary_op(self):
        return self._summary_op
    @property
    def train_op(self):
        return self._train_op
    @property
    def reset_initState_op(self):
        return self._reset_init_state_op
    @property
    def transfer_initState_op(self):
        return self._transfer_state_op
    @property
    def loss(self):
        return self._loss
    @property
    def accuracy(self):
        return self._accuracy
    @property
    def learning_rate(self):
        return self._learning_rate

