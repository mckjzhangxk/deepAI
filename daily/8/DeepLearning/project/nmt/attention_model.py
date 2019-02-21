# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Attention-based sequence-to-sequence model with dynamic RNN support."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from . import model
from . import model_helper

__all__ = ["AttentionModel"]


class AttentionModel(model.Model):
  """Sequence-to-sequence dynamic model with attention.

  This class implements a multi-layer recurrent neural network as encoder,
  and an attention-based decoder. This is the same as the model described in
  (Luong et al., EMNLP'2015) paper: https://arxiv.org/pdf/1508.04025v5.pdf.
  This class also allows to use GRU cells in addition to LSTM cells with
  support for dropout.
  """

  def __init__(self,
               hparams,
               mode,
               iterator,
               source_vocab_table,
               target_vocab_table,
               reverse_target_vocab_table=None,
               scope=None,
               extra_args=None):
    self.has_attention = hparams.attention_architecture and hparams.attention

    # Set attention_mechanism_fn
    if self.has_attention:
      if extra_args and extra_args.attention_mechanism_fn:
        self.attention_mechanism_fn = extra_args.attention_mechanism_fn
      else:
        self.attention_mechanism_fn = create_attention_mechanism

    super(AttentionModel, self).__init__(
        hparams=hparams,
        mode=mode,
        iterator=iterator,
        source_vocab_table=source_vocab_table,
        target_vocab_table=target_vocab_table,
        reverse_target_vocab_table=reverse_target_vocab_table,
        scope=scope,
        extra_args=extra_args)
  '''
  memory:(batch,T,D)
  encoder_state:((C0,h0)....(cn,hn)),ci=(batch,dim_i)
  source_sequence_length:(batch,)
  
  把memory,encoder_state,source_sequence_length的第一维度变成batch_size * beam_width
  '''
  def _prepare_beam_search_decoder_inputs(
      self, beam_width, memory, source_sequence_length, encoder_state):
    memory = tf.contrib.seq2seq.tile_batch(
        memory, multiplier=beam_width)
    source_sequence_length = tf.contrib.seq2seq.tile_batch(
        source_sequence_length, multiplier=beam_width)
    encoder_state = tf.contrib.seq2seq.tile_batch(
        encoder_state, multiplier=beam_width)
    batch_size = self.batch_size * beam_width
    return memory, source_sequence_length, encoder_state, batch_size
    '''
    
    
    cell:可以看作是一个垂直的rnn单元,最上包裹了attention layer,例如
        lstm1->lstm2->...lstm_n->attention_layer
        这个cell被调用,输出应该是attention_vector(N,T,numdim),hparams.output_attention=true
        和final_state:每一层的lstm的最终状态(ci,hi),ci~(batch,Di)
    decoder_initial_state:
        cell.zero_state(batch_size, dtype),输入给decoder的初始状态,
        对于beam search,batch_size=batch_size*bw
    
    '''
  def _build_decoder_cell(self, hparams, encoder_outputs, encoder_state,
                          source_sequence_length):
    """Build a RNN cell with attention mechanism that can be used by decoder."""
    # No Attention
    if not self.has_attention:
      return super(AttentionModel, self)._build_decoder_cell(
          hparams, encoder_outputs, encoder_state, source_sequence_length)
    elif hparams.attention_architecture != "standard":
      raise ValueError(
          "Unknown attention architecture %s" % hparams.attention_architecture)
    #有隐藏层,atten_layer的维度
    num_units = hparams.num_units
    #层数
    num_layers = self.num_decoder_layers
    num_residual_layers = self.num_decoder_residual_layers
    infer_mode = hparams.infer_mode

    dtype = tf.float32

    # Ensure memory is batch-major
    if self.time_major:
      memory = tf.transpose(encoder_outputs, [1, 0, 2])
    else:
      memory = encoder_outputs
    #memory=(batch,T,D)
    if (self.mode == tf.contrib.learn.ModeKeys.INFER and
        infer_mode == "beam_search"):
      memory, source_sequence_length, encoder_state, batch_size = (
          self._prepare_beam_search_decoder_inputs(
              hparams.beam_width, memory, source_sequence_length,
              encoder_state))
    else:
      batch_size = self.batch_size

    # Attention,attention_mechanism代表如何计算content_vector,已经attention layer输出的维度
    attention_mechanism = self.attention_mechanism_fn(
        hparams.attention, num_units, memory, source_sequence_length, self.mode)
    #表示创建了num_layers层,每层使用unit_type单元,每层维度是num_units
    cell = model_helper.create_rnn_cell(
        unit_type=hparams.unit_type,
        num_units=num_units,
        num_layers=num_layers,
        num_residual_layers=num_residual_layers,
        forget_bias=hparams.forget_bias,
        dropout=hparams.dropout,
        num_gpus=self.num_gpus,
        mode=self.mode,
        single_cell_fn=self.single_cell_fn)

    # Only generate alignment in greedy INFER mode.
    alignment_history = (self.mode == tf.contrib.learn.ModeKeys.INFER and
                         infer_mode != "beam_search")
    #可以理解成在最高层插入了attention层
    '''
    attention_layer_size:attention输出的维度,None的话输出context_value(也就是是memory的维度一样),
    设置后,把cell.output和context_value联合起来,转化成attention_layer_size的向量
    at=tanh(W[ct,ht])
    '''
    cell = tf.contrib.seq2seq.AttentionWrapper(
        cell,
        attention_mechanism,
        attention_layer_size=num_units,
        alignment_history=alignment_history,
        output_attention=hparams.output_attention,
        name="attention")

    # TODO(thangluong): do we need num_layers, num_gpus?
    cell = tf.contrib.rnn.DeviceWrapper(cell,
                                        model_helper.get_device_str(
                                            num_layers - 1, self.num_gpus))
    #是否把encoder的输出给decoder
    if hparams.pass_hidden_state:
      decoder_initial_state = cell.zero_state(batch_size, dtype).clone(
          cell_state=encoder_state)
    else:
      decoder_initial_state = cell.zero_state(batch_size, dtype)

    return cell, decoder_initial_state

  def _get_infer_summary(self, hparams):
    if not self.has_attention or hparams.infer_mode == "beam_search":
      return tf.no_op()
    return _create_attention_images_summary(self.final_context_state)

'''
attention_option:string luong | scaled_luong | bahdanau | normed_bahdanau
num_units:hidden layers dims:default 128
memory:(N,T,Dlast_en) encoder最后一层的输出
source_sequence_length:(N,)源长度
mode:TRAIN|EVAL|INFER

返回:
tf.contrib.seq2seq.LuongAttention:score(ht,hs)=htWshs
tf.contrib.seq2seq.BahdanauAttention score(ht,hs)=v_a W[ht;hs]
    
    LuongAttention(num_units,memory,memory_sequence):
    
    定义LuongAttention,表示:
        1.将来作为query的ht,维度是num_units,
        2.encoder给出的memory.shape是(N,T,hs)
        3.参与计算score的是头memory[:memory_sequence]
        4.为了和ht尺度兼容,要定义一个Dense(num_units),把memory转成keys,
            keys.shape=(N,T,ht)
        5.当LuongAttention被调用的时候,传入的query应该是(N,ht)维度的,
        这个函数做的是score=keys.dot(query)=(N,T),然后得到
        alignments=softmax(score),表示query对memory[0,T)部分的官渡程度
        alignments.shape=(N,T)
'''
def create_attention_mechanism(attention_option, num_units, memory,
                               source_sequence_length, mode):
  """Create attention mechanism based on the attention_option."""
  del mode  # unused

  # Mechanism
  if attention_option == "luong":
    attention_mechanism = tf.contrib.seq2seq.LuongAttention(
        num_units, memory, memory_sequence_length=source_sequence_length)
  elif attention_option == "scaled_luong":
    attention_mechanism = tf.contrib.seq2seq.LuongAttention(
        num_units,
        memory,
        memory_sequence_length=source_sequence_length,
        scale=True)
  elif attention_option == "bahdanau":
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
        num_units, memory, memory_sequence_length=source_sequence_length)
  elif attention_option == "normed_bahdanau":
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
        num_units,
        memory,
        memory_sequence_length=source_sequence_length,
        normalize=True)
  else:
    raise ValueError("Unknown attention option %s" % attention_option)

  return attention_mechanism


def _create_attention_images_summary(final_context_state):
  """create attention image and attention summary."""
  attention_images = (final_context_state.alignment_history.stack())
  # Reshape to (batch, src_seq_len, tgt_seq_len,1)
  attention_images = tf.expand_dims(
      tf.transpose(attention_images, [1, 2, 0]), -1)
  # Scale to range [0, 255]
  attention_images *= 255
  attention_summary = tf.summary.image("attention_images", attention_images)
  return attention_summary
