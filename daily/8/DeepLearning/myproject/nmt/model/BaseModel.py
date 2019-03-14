import tensorflow as tf
import  modelHelper as helper
from utils.vocab_utils import get_special_word_id,check_vocab
import collections
import numpy as np
from utils.nmt_utils import get_translation

class TrainOutput(collections.namedtuple('TrainOutput',['loss','lr','summary','global_norm','clip_global_norm','word_count','predict_count','global_step'])):pass
class InferOutput(collections.namedtuple('InferOutput',['translation'])):pass

class BaseModel():
    def __init__(self,batchInput,mode,hparam):
        self._batchInput = batchInput
        self.mode=mode.lower()
        self._set_commom_param(hparam)

        with tf.variable_scope(hparam.scope):
            self._buildNetwork(hparam)
            self._set_Saver(hparam)
    def _set_commom_param(self,hparam):
        self._batch=tf.shape(self._batchInput.src)[0]
        self.C,_=check_vocab(hparam.vocab_tgt)
        self.SOS=hparam.SOS
        self.EOS=hparam.EOS
        self._subword=hparam.subword_option
        if self.mode!='infer':
            self._predict_count=tf.reduce_sum(self._batchInput.tgt_seq_len)
            self._word_count=tf.reduce_sum(self._batchInput.src_seq_len)+tf.reduce_sum(self._batchInput.tgt_seq_len)


    def _build_emb_layer(self,hparam):
        '''
        根据hparam的vocab_src,vocab_tgt,share_vocab
        分别创建encoder和decoder的emb varibale
        然后把self.batch.src用encoder_variable编码
        
        如果mode不是eval
            把self.batch.src用decoder_variable编码
        否在:
            保存decoder_variable
        模型新增两个成员
            _emb_src,emb_tgt
        :param hparam: 
        :return: 
        '''
        emb_encoder,emb_decoder=helper.create_emb_matric(hparam)

        self._emb_src = tf.nn.embedding_lookup(emb_encoder, self._batchInput.src)
        if self.mode!='infer':
            self._emb_tgt=tf.nn.embedding_lookup(emb_decoder, self._batchInput.tgt_inp)
        else:
            self._emb_tgt=emb_decoder
    def _build_encoder_cell(self,hparam):
        '''
        根据hparam创建rnn cell,
        
        返回:
            对于多层网络返回
                MultiRnnCell
            对于单层网络返回:
                BaseRnnCell
        :param hparam: 
        :return: 
        '''
        dropout=hparam.dropout if self.mode=='train' else 0.0
        rnn_cell=helper.create_rnn_cell(hparam,dropout)
        return rnn_cell

    def _build_encoder(self,hparam):
        '''
        创建成员
            self._encoder_state:tutle_(?,T,ndim,float32)
            self._encoder_encoder:(?,T,ndim,float32),这里的
            输出不一定是最后rnn的最后一个单元输出，而是rnn的src_seq_len单元
        :param hparam: 
        :return: 
        '''
        if hparam.encode_type=='uni':
            rnn_cell=self._build_encoder_cell(hparam)
            self._encoder_output,self._encode_state=\
                tf.nn.dynamic_rnn(rnn_cell,
                                  self._emb_src,
                                  sequence_length=self._batchInput.src_seq_len,
                                  dtype=tf.float32)
        elif hparam.encode_type=='bi':
            fw_cell = self._build_encoder_cell(hparam)
            bk_cell = self._build_encoder_cell(hparam)

            _encoder_output,_encode_state=\
                tf.nn.bidirectional_dynamic_rnn(
                                            fw_cell,
                                            bk_cell,
                                            self._emb_src,
                                            dtype=tf.float32)
            self._encode_state=_encode_state[0]
            self._encoder_output=tf.concat(_encoder_output,-1)
    def _build_decoder(self,hparam):
        raise NotImplementedError

    def _buildNetwork(self,hparam):
        self._build_emb_layer(hparam)

        self._build_encoder(hparam)
        self._build_decoder(hparam)

        if self.mode!='infer':
            self._set_train_eval(hparam)
        # if self.mode!='train':
        #     se
    def _set_decay_lr(self,hparam):
        '''
            调用之前self.lr,self.global_step
            已经设置好了
        :param hparam: 
        :return: 
        '''
        decay_scheme=hparam.decay_scheme
        num_train=hparam.num_train

        if decay_scheme is None:return

        if decay_scheme=='luong234':
            begin_step=int(num_train*2/3)
            decay_step=int(num_train/3/4)
        if decay_scheme == 'luong5':
            begin_step = int(num_train / 2)
            decay_step = int(num_train / 2 / 5)
        if decay_scheme == 'luong10':
            begin_step = int(num_train / 2)
            decay_step = int(num_train / 2 / 10)
        base_lr=self.lr
        self.lr=tf.cond(self.global_step<begin_step,
                        lambda :base_lr,
                        lambda :tf.train.exponential_decay(base_lr,self.global_step-begin_step,decay_step,0.5,True)
                        )

    def _warmup_lr(self,hparam):
        if hparam.warmup_scheme=='t2t':
            warmup_factor=tf.exp(tf.log(0.01)/hparam.warmup_steps)
            inv_decay=tf.pow(warmup_factor, tf.to_float(hparam.warmup_steps-self.global_step))

        base_lr=self.lr
        self.lr=tf.cond(self.global_step<hparam.warmup_steps,
                        lambda :base_lr*inv_decay,
                        lambda :base_lr)
    def _set_lr(self,hparam):
        self.lr=hparam.lr
        self._warmup_lr(hparam)
        self._set_decay_lr(hparam)

    def _set_loss(self):
        #(?,T)
        loss=tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self._logit,
            labels=self._batchInput.tgt_out)
        mask=tf.to_float(tf.sequence_mask(self._batchInput.tgt_seq_len,tf.reduce_max(self._batchInput.tgt_seq_len)))
        #(?,)
        loss=tf.reduce_sum(loss*mask,axis=1)
        self._loss=tf.reduce_mean(loss)

    def _set_train_summary(self,hparam):
        tf.summary.scalar('loss',self._loss)
        tf.summary.scalar('global_norm',self._global_norm)
        tf.summary.scalar('clip_global_norm', self._clip_global_norm)
        tf.summary.scalar('lr',self.lr)
        self._summary=tf.summary.merge_all()

    def _set_train_eval(self, hparam):
        '''
        设置train和eval的目标，
        train要设置
            loss,solver,lr,clip,gradient,以及summary
        eval要设置
            loss
        备注:
            train的loss和eval的loss 并不相同,
            train.loss表示批次的平均
            eval.loss表示批次的总,用于perplexity
        :param hparam: 
        :return: 
        '''
        self._set_loss()
        if self.mode=='train':
            self.global_step = tf.Variable(0, trainable=False, name='global_step',dtype=tf.int32)
            self._set_lr(hparam)

            #设置优化算法
            if hparam.optimizer=='adam':
                optimizer=tf.train.AdamOptimizer(self.lr)
            elif hparam.optimizer=='sgd':
                optimizer = tf.train.GradientDescentOptimizer(self.lr)
            else:
                raise ValueError('Unknown Optimizer %s'%hparam.optimizer)
            vars=tf.trainable_variables()
            grads=tf.gradients(self._loss,vars)
            clip_grads,self._global_norm=tf.clip_by_global_norm(grads,hparam.max_norm)
            self._clip_global_norm=tf.global_norm(clip_grads)
            self._train_op=optimizer.apply_gradients(zip(clip_grads, vars), self.global_step)

            self._set_train_summary(hparam)
        else:
            self._loss=self._loss*tf.to_float(self._batch)

    def _set_Saver(self,hparam):
        varlist=tf.global_variables()

        self._saver=tf.train.Saver(var_list=varlist,max_to_keep=hparam.max_to_keep)
    def save(self,sess,model_dir,global_step=None):
        self._saver.save(sess,model_dir,global_step)
    def restore(self,sess,model_dir):
        self._saver.restore(sess,model_dir)

    def train(self,sess):
        op=[self._train_op,
            self._loss,
            self.lr,
            self._summary,
            self._global_norm,
            self._clip_global_norm,
            self._word_count,
            self._predict_count,
            self.global_step]

        op_result=sess.run(op)
        # 'loss', 'lr', 'summary', 'global_norm','clip_global_norm', 'word_count', 'predict_count,global_step'
        return TrainOutput(*op_result[1:])

    def eval(self,sess):
        '''
        做eval,返回测试集合的perplexity
        :param sess: 
        :return: 
        '''
        total_loss,predict_cnt=0,0
        sess.run(self._batchInput.initializer)
        while True:
            try:
                _loss,_predict_cnt=sess.run([self._loss,self._predict_count])
                total_loss+=_loss
                predict_cnt+=_predict_cnt
            except tf.errors.OutOfRangeError:
                break
        return np.exp(total_loss/predict_cnt)
    def infer(self,sess):
        '''
        对输入进行翻译，返回翻译的结果
        
        :param sess: 
        :return: 
        '''
        sess.run(self._batchInput.initializer)
        translation=[]
        while True:
            try:
                _translation=sess.run(self._result)
                if self._isBeam:
                    #选择beam search得分最高的
                    _translation=_translation[:,:,0]

                for i,_ in enumerate(_translation):
                    translation.append(get_translation(_translation,i,self.EOS,self._subword))

            except tf.errors.OutOfRangeError:
                break
        return InferOutput(translation)
class Model(BaseModel):
    def __init__(self,batch_inp,mode,hparam):
        super(Model,self).__init__(batch_inp,mode,hparam)

    def _build_decode_cell(self,hparam):
        '''
        创建与encoder，一模一样的结构，返回
        rnn_cell,decode_init_state
        :param hparam: 
        :return: 
        '''
        dropout=hparam.dropout if hparam=='train' else  0.0
        rnn_cell=helper.create_rnn_cell(hparam,dropout)
        init_state=self._encode_state
        if self.mode=='infer' and hparam.infer_mode=='beam_search':
            init_state=tf.contrib.seq2seq.tile_batch(init_state,hparam.beam_width)
        return rnn_cell,init_state
    def __max__iteration(self):
        return tf.round(tf.reduce_max(self._batchInput.src_seq_len)*2)
    def __set_decoder_output__(self, rnn_cell, init_state, hparam):
        '''
        根据RNN单元和RNN的初始状态，得到decoder的输出，
        输出主要包括2部分，
            self.logit:(?,T,D,float32):针对train,eval
            self.sampleid:(?,T,int32):针对eval,infer
            对于beamsearch,:(?,T,bw,int32)
        :param rnn_cell: 
        :param init_state: 
        :return: 
        '''
        proj = tf.layers.Dense(self.C)
        if self.mode!='infer':
            helper=tf.contrib.seq2seq.TrainingHelper(self._emb_tgt,
                                                     self._batchInput.tgt_seq_len)
            decoder=tf.contrib.seq2seq.BasicDecoder(rnn_cell,helper,init_state,output_layer=proj)
            final_outputs,_,_=tf.contrib.seq2seq.dynamic_decode(decoder,output_time_major=False)

            self._logit=final_outputs.rnn_output
            self._sample_id=final_outputs.sample_id
        elif hparam.infer_mode!='beam_search':
            tgt_sos_id, tgt_eos_id = get_special_word_id(hparam)
            start_tokens = tf.fill([self._batch], tgt_sos_id)

            if hparam.infer_mode=='sample':
                helper = tf.contrib.seq2seq.SampleEmbeddingHelper(
                    self._emb_tgt,
                    start_tokens=start_tokens,
                    end_token=tgt_eos_id,
                    softmax_temperature=1.0
                )
            elif hparam.infer_mode=='greedy':
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    self._emb_tgt,
                    start_tokens=start_tokens,
                    end_token=tgt_eos_id)
            decoder = tf.contrib.seq2seq.BasicDecoder(rnn_cell, helper, init_state,output_layer=proj)
            final_outputs,_,_=tf.contrib.seq2seq.dynamic_decode(decoder,
                                                               output_time_major = False,
                                                               maximum_iterations=self.__max__iteration())
            self._isBeam = False
            self._logit=tf.no_op()
            self._sample_id=final_outputs.sample_id
            self._result=self._batchInput.reverse_vocab.lookup(tf.to_int64(self._sample_id))

        else:
            tgt_sos_id, tgt_eos_id = get_special_word_id(hparam)
            start_tokens = tf.fill([self._batch], tgt_sos_id)

            decoder=tf.contrib.seq2seq.BeamSearchDecoder(rnn_cell,self._emb_tgt,
                                                 start_tokens=start_tokens,
                                                 end_token=tgt_eos_id,
                                                 initial_state=init_state,
                                                 beam_width=hparam.beam_width,
                                                 output_layer=proj)
            final_outputs,_,_ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                  maximum_iterations=self.__max__iteration(),
                                                                  output_time_major=False)

            self._isBeam=True
            self._logit=tf.no_op()
            self._sample_id=final_outputs.predicted_ids
            self._result = self._batchInput.reverse_vocab.lookup(tf.to_int64(self._sample_id))

    def _build_decoder(self,hparam):

        rnn_cell,decode_init_state=self._build_decode_cell(hparam)

        self.__set_decoder_output__(rnn_cell, decode_init_state, hparam)