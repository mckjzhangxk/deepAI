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
"""For training NMT models."""
from __future__ import print_function

import math
import os
import random
import time

import tensorflow as tf

from . import attention_model
from . import gnmt_model
from . import inference
from . import model as nmt_model
from . import model_helper
from .utils import misc_utils as utils
from .utils import nmt_utils

utils.check_tensorflow_version()

__all__ = [
    "run_sample_decode", "run_internal_eval", "run_external_eval",
    "run_avg_external_eval", "run_full_eval", "init_stats", "update_stats",
    "print_step_info", "process_stats", "train", "get_model_creator",
    "add_info_summaries", "get_best_results"
]

'''
model_dir:模型参数保存的路径
infer_model:
infer_sess:用于运行infer的session
src_data:list,每个元素是一句,char类型
tgt_data:list,每个元素是一句,char类型
summary_writer:用于记录train的writer

使用src_data,执行一次infer(把src_data翻译成目标语言),
与tgt_data对比

打印出
    src_data[chioceid]
    tgt_data[chioceid]
    nmt(src_data[chioceid])
'''

def run_sample_decode(infer_model, infer_sess, model_dir, hparams,
                      summary_writer, src_data, tgt_data):
  """Sample decode a random sentence from src_data."""
  with infer_model.graph.as_default():
    '''
    这一步实际上是从model_dir恢复网络的参数,网络在创建infer_model已经创建好啦,
    如果model_dir没有参数,那么就run(global_initializer)
    '''
    loaded_infer_model, global_step = model_helper.create_or_load_model(
        infer_model.model, model_dir, infer_sess, "infer")
  '''
    loaded_infer_model:model.Model,tensorflow中定义的infer计算图,已经'初始化'完成
    infer_sess:
    infer_model.iterator
    infer_model.src_placeholder--------->|                  |
                                         |get_infer_iterator|-->infer_model.iterator
    infer_model.batch_size_placeholder-->|                  |
    上3个参数定义了infer计算图的输入
    src_data, tgt_data:是实际输入的数据
    
  '''
  _sample_decode(loaded_infer_model, global_step, infer_sess, hparams,
                 infer_model.iterator, src_data, tgt_data,
                 infer_model.src_placeholder,
                 infer_model.batch_size_placeholder, summary_writer)

'''

对dev集,test集,计算perplexity,dev,并且保存到日志文件里面,test集都是hparams给出
返回
(dev_ppl,test_ppl)

'''
def run_internal_eval(eval_model,
                      eval_sess,
                      model_dir,
                      hparams,
                      summary_writer,
                      use_test_set=True,
                      dev_eval_iterator_feed_dict=None,
                      test_eval_iterator_feed_dict=None):
  """Compute internal evaluation (perplexity) for both dev / test.

  Computes development and testing perplexities for given model.

  Args:
    eval_model: Evaluation model for which to compute perplexities.
    eval_sess: Evaluation TensorFlow session.
    model_dir: Directory from which to load evaluation model from.
    hparams: Model hyper-parameters.
    summary_writer: Summary writer for logging metrics to TensorBoard.
    use_test_set: Computes testing perplexity if true; does not otherwise.
      Note that the development perplexity is always computed regardless of
      value of this parameter.
    dev_eval_iterator_feed_dict: Feed dictionary for a TensorFlow session.
      Can be used to pass in additional inputs necessary for running the
      development evaluation.
    test_eval_iterator_feed_dict: Feed dictionary for a TensorFlow session.
      Can be used to pass in additional inputs necessary for running the
      testing evaluation.
  Returns:
    Pair containing development perplexity and testing perplexity, in this
    order.
  """
  if dev_eval_iterator_feed_dict is None:
    dev_eval_iterator_feed_dict = {}
  if test_eval_iterator_feed_dict is None:
    test_eval_iterator_feed_dict = {}
  with eval_model.graph.as_default():
    loaded_eval_model, global_step = model_helper.create_or_load_model(
        eval_model.model, model_dir, eval_sess, "eval")

  dev_src_file = "%s.%s" % (hparams.dev_prefix, hparams.src)
  dev_tgt_file = "%s.%s" % (hparams.dev_prefix, hparams.tgt)
  '''
    EvaluateInterator(graph,model,interator,src_file_place,tgt_file_placeholder)
    
   src_file_place--------->|                           |
                           |iterator_utils.get_iterator|-->interator
   tgt_file_placeholder--->|                           |
   
  '''
  dev_eval_iterator_feed_dict[eval_model.src_file_placeholder] = dev_src_file
  dev_eval_iterator_feed_dict[eval_model.tgt_file_placeholder] = dev_tgt_file
  '''
  loaded_eval_model:eval 模型的抽象计算图
  eval_sess:
  eval_model.iterator:eval 模型计算图的抽象输入
  dev_eval_iterator_feed_dict:计算图的实际输入
  global_step
  '''
  dev_ppl = _internal_eval(loaded_eval_model, global_step, eval_sess,
                           eval_model.iterator, dev_eval_iterator_feed_dict,
                           summary_writer, "dev")
  test_ppl = None
  if use_test_set and hparams.test_prefix:
    test_src_file = "%s.%s" % (hparams.test_prefix, hparams.src)
    test_tgt_file = "%s.%s" % (hparams.test_prefix, hparams.tgt)
    test_eval_iterator_feed_dict[
        eval_model.src_file_placeholder] = test_src_file
    test_eval_iterator_feed_dict[
        eval_model.tgt_file_placeholder] = test_tgt_file
    test_ppl = _internal_eval(loaded_eval_model, global_step, eval_sess,
                              eval_model.iterator, test_eval_iterator_feed_dict,
                              summary_writer, "test")
  return dev_ppl, test_ppl

'''
使用infer_model,对dev,test数据集得出hparams.metrics指标,
当模型参数指标最好(save_best_dev)的时候保存ckpt,到best_+{best_metric_label}+_dir下面translate.ckpt

#运行一次run_external_eval,保存最好的指标模型到hparams.best_metric_dir下面,
各个指标的计算结果保存到summary_writer.Summary($(dev/test)_metric,metric_score,global_step中)下面,
使用tensorboard可以进行查看,对dev/test翻译的结果保存到hparams.outdir/output_dev或output_test下

返回:dev_scores, test_scores, global_step
dev_scores,test_scores:{metrics_name:metrics_score}
global_step:是当前运行次数(int)
'''

def run_external_eval(infer_model,
                      infer_sess,
                      model_dir,
                      hparams,
                      summary_writer,
                      save_best_dev=True,
                      use_test_set=True,
                      avg_ckpts=False,
                      dev_infer_iterator_feed_dict=None,
                      test_infer_iterator_feed_dict=None):
  """Compute external evaluation for both dev / test.

  Computes development and testing external evaluation (e.g. bleu, rouge) for
  given model.

  Args:
    infer_model: Inference model for which to compute perplexities.
    infer_sess: Inference TensorFlow session.
    model_dir: Directory from which to load inference model from.
    hparams: Model hyper-parameters.
    summary_writer: Summary writer for logging metrics to TensorBoard.
    use_test_set: Computes testing external evaluation if true; does not
      otherwise. Note that the development external evaluation is always
      computed regardless of value of this parameter.
    dev_infer_iterator_feed_dict: Feed dictionary for a TensorFlow session.
      Can be used to pass in additional inputs necessary for running the
      development external evaluation.
    test_infer_iterator_feed_dict: Feed dictionary for a TensorFlow session.
      Can be used to pass in additional inputs necessary for running the
      testing external evaluation.
  Returns:
    Triple containing development scores, testing scores and the TensorFlow
    Variable for the global step number, in this order.
  """
  if dev_infer_iterator_feed_dict is None:
    dev_infer_iterator_feed_dict = {}
  if test_infer_iterator_feed_dict is None:
    test_infer_iterator_feed_dict = {}
  with infer_model.graph.as_default():
    loaded_infer_model, global_step = model_helper.create_or_load_model(
        infer_model.model, model_dir, infer_sess, "infer")

  dev_src_file = "%s.%s" % (hparams.dev_prefix, hparams.src)
  dev_tgt_file = "%s.%s" % (hparams.dev_prefix, hparams.tgt)
  '''
  设在好了输入,注意不同于eval_internal,这里值设在了源翻译文件,和infer_batch_size=32(default)
  '''
  dev_infer_iterator_feed_dict[
      #src_placeholder是一个字符串数组的占位符,所有这里把dev文件全部加载了过来(一个字符串数组)
      # ,feed into src_placeholder
      infer_model.src_placeholder] = inference.load_data(dev_src_file)
  dev_infer_iterator_feed_dict[
      infer_model.batch_size_placeholder] = hparams.infer_batch_size
  dev_scores = _external_eval(
      loaded_infer_model,
      global_step,
      infer_sess,
      hparams,
      infer_model.iterator,
      dev_infer_iterator_feed_dict,
      dev_tgt_file,
      "dev",
      summary_writer,
      save_on_best=save_best_dev,
      avg_ckpts=avg_ckpts)

  test_scores = None
  if use_test_set and hparams.test_prefix:
    test_src_file = "%s.%s" % (hparams.test_prefix, hparams.src)
    test_tgt_file = "%s.%s" % (hparams.test_prefix, hparams.tgt)
    test_infer_iterator_feed_dict[
        infer_model.src_placeholder] = inference.load_data(test_src_file)
    test_infer_iterator_feed_dict[
        infer_model.batch_size_placeholder] = hparams.infer_batch_size
    test_scores = _external_eval(
        loaded_infer_model,
        global_step,
        infer_sess,
        hparams,
        infer_model.iterator,
        test_infer_iterator_feed_dict,
        test_tgt_file,
        "test",
        summary_writer,
        save_on_best=False,
        avg_ckpts=avg_ckpts)
  return dev_scores, test_scores, global_step
'''
global_step:int,当前已经运行的步数
把过去checkpoint保存的变量做平均,然后使用平均值做external_eval
'''
def run_avg_external_eval(infer_model, infer_sess, model_dir, hparams,
                          summary_writer, global_step):
  """Creates an averaged checkpoint and run external eval with it."""
  avg_dev_scores, avg_test_scores = None, None
  if hparams.avg_ckpts:
    # Convert VariableName:0 to VariableName.
    global_step_name = infer_model.model.global_step.name.split(":")[0]
    avg_model_dir = model_helper.avg_checkpoints(
        model_dir, hparams.num_keep_ckpts, global_step, global_step_name)

    if avg_model_dir:
      avg_dev_scores, avg_test_scores, _ = run_external_eval(
          infer_model,
          infer_sess,
          avg_model_dir,
          hparams,
          summary_writer,
          avg_ckpts=True)

  return avg_dev_scores, avg_test_scores
'''
使用eval_model计算dev,test数据集的ppl
使用infer_model计算dev,test数据集的metrics[blue,rough,accuray]
并且会把结果计入日志(summary_writer,global_step)

并且会把每个best_metrics模型的参数保存在:best_metricsname_dir/translate.cpkt文件下面
返回:
result_summary:string:
    dev ppl 22,dev blue 33,dev accuracy 0.1,
    test ppl.....
    
global_step:当前次数
metrics:分数结果
  {
      "dev_ppl": dev_ppl,
      "test_ppl": test_ppl,
      "dev_scores": dev_scores,
      "test_scores": test_scores,
  }
'''

def run_internal_and_external_eval(model_dir,
                                   infer_model,
                                   infer_sess,
                                   eval_model,
                                   eval_sess,
                                   hparams,
                                   summary_writer,
                                   avg_ckpts=False,
                                   dev_eval_iterator_feed_dict=None,
                                   test_eval_iterator_feed_dict=None,
                                   dev_infer_iterator_feed_dict=None,
                                   test_infer_iterator_feed_dict=None):
  """Compute internal evaluation (perplexity) for both dev / test.

  Computes development and testing perplexities for given model.

  Args:
    model_dir: Directory from which to load models from.
    infer_model: Inference model for which to compute perplexities.
    infer_sess: Inference TensorFlow session.
    eval_model: Evaluation model for which to compute perplexities.
    eval_sess: Evaluation TensorFlow session.
    hparams: Model hyper-parameters.
    summary_writer: Summary writer for logging metrics to TensorBoard.
    avg_ckpts: Whether to compute average external evaluation scores.
    dev_eval_iterator_feed_dict: Feed dictionary for a TensorFlow session.
      Can be used to pass in additional inputs necessary for running the
      internal development evaluation.
    test_eval_iterator_feed_dict: Feed dictionary for a TensorFlow session.
      Can be used to pass in additional inputs necessary for running the
      internal testing evaluation.
    dev_infer_iterator_feed_dict: Feed dictionary for a TensorFlow session.
      Can be used to pass in additional inputs necessary for running the
      external development evaluation.
    test_infer_iterator_feed_dict: Feed dictionary for a TensorFlow session.
      Can be used to pass in additional inputs necessary for running the
      external testing evaluation.
  Returns:
    Triple containing results summary, global step Tensorflow Variable and
    metrics in this order.
  """

  #使用EVAL_model,计算dev,test的perplexity
  dev_ppl, test_ppl = run_internal_eval(
      eval_model,
      eval_sess,
      model_dir,
      hparams,
      summary_writer,
      dev_eval_iterator_feed_dict=dev_eval_iterator_feed_dict,
      test_eval_iterator_feed_dict=test_eval_iterator_feed_dict)
  # 使用Infer_model,计算dev,test的metrics_scores
  dev_scores, test_scores, global_step = run_external_eval(
      infer_model,
      infer_sess,
      model_dir,
      hparams,
      summary_writer,
      dev_infer_iterator_feed_dict=dev_infer_iterator_feed_dict,
      test_infer_iterator_feed_dict=test_infer_iterator_feed_dict)

  metrics = {
      "dev_ppl": dev_ppl,
      "test_ppl": test_ppl,
      "dev_scores": dev_scores,
      "test_scores": test_scores,
  }

  avg_dev_scores, avg_test_scores = None, None
  if avg_ckpts:
    avg_dev_scores, avg_test_scores = run_avg_external_eval(
        infer_model, infer_sess, model_dir, hparams, summary_writer,
        global_step)
    metrics["avg_dev_scores"] = avg_dev_scores
    metrics["avg_test_scores"] = avg_test_scores

  result_summary = _format_results("dev", dev_ppl, dev_scores, hparams.metrics)
  if avg_dev_scores:
    result_summary += ", " + _format_results("avg_dev", None, avg_dev_scores,
                                             hparams.metrics)
  if hparams.test_prefix:
    result_summary += ", " + _format_results("test", test_ppl, test_scores,
                                             hparams.metrics)
    if avg_test_scores:
      result_summary += ", " + _format_results("avg_test", None,
                                               avg_test_scores, hparams.metrics)

  return result_summary, global_step, metrics

'''
model_dir:模型参数保存的路径
infer_model:
infer_sess:用于运行infer的session
eval_model:
eval_sess:用于运行eval的session
sample_src_data:list,每个元素是一句,char类型
sample_tgt_data:list,每个元素是一句,char类型
summary_writer:用于记录train的writer

1.随机翻译sample_src_data的句子,打印到控制台
2.计算dev,test数据集的ppl
3.计算dev,test数据集的metrics[blue,rough,accuray]
4.并且会把结果计入日志(summary_writer,global_step)


返回:
result_summary:string:
    dev ppl 22,dev blue 33,dev accuracy 0.1,
    test ppl.....
    
global_step:当前次数
metrics:分数结果
  {
      "dev_ppl": dev_ppl,
      "test_ppl": test_ppl,
      "dev_scores": dev_scores,
      "test_scores": test_scores,
  }
'''
def run_full_eval(model_dir,
                  infer_model,
                  infer_sess,
                  eval_model,
                  eval_sess,
                  hparams,
                  summary_writer,
                  sample_src_data,
                  sample_tgt_data,
                  avg_ckpts=False):
  """Wrapper for running sample_decode, internal_eval and external_eval.

  Args:
    model_dir: Directory from which to load models from.
    infer_model: Inference model for which to compute perplexities.
    infer_sess: Inference TensorFlow session.
    eval_model: Evaluation model for which to compute perplexities.
    eval_sess: Evaluation TensorFlow session.
    hparams: Model hyper-parameters.
    summary_writer: Summary writer for logging metrics to TensorBoard.
    sample_src_data: sample of source data for sample decoding.
    sample_tgt_data: sample of target data for sample decoding.
    avg_ckpts: Whether to compute average external evaluation scores.
  Returns:
    Triple containing results summary, global step Tensorflow Variable and
    metrics in this order.
  """
  run_sample_decode(infer_model, infer_sess, model_dir, hparams, summary_writer,
                    sample_src_data, sample_tgt_data)
  return run_internal_and_external_eval(model_dir, infer_model, infer_sess,
                                        eval_model, eval_sess, hparams,
                                        summary_writer, avg_ckpts)


def init_stats():
  """Initialize statistics that we want to accumulate."""
  return {"step_time": 0.0, "train_loss": 0.0,
          "predict_count": 0.0,  # word count on the target side
          "word_count": 0.0,  # word counts for both source and target
          "sequence_count": 0.0,  # number of training examples processed
          "grad_norm": 0.0}
'''
更新states:
    step_time:目前运行用时(s)
    train_loss:累计loss
    grad_norm:累计global_norm
    predict_count:累计目标单词数量
    word_count:累计目标单词数量+源单词数量
    sequence_count:累计遍历的句子
返回:
    (global_step,learning_rate,train_summary)
'''

def update_stats(stats, start_time, step_result):
  """Update stats: write summary and accumulate statistics."""
  _, output_tuple = step_result

  # Update statistics
  batch_size = output_tuple.batch_size
  stats["step_time"] += time.time() - start_time
  stats["train_loss"] += output_tuple.train_loss * batch_size
  stats["grad_norm"] += output_tuple.grad_norm
  stats["predict_count"] += output_tuple.predict_count
  stats["word_count"] += output_tuple.word_count
  stats["sequence_count"] += batch_size

  return (output_tuple.global_step, output_tuple.learning_rate,
          output_tuple.train_summary)


def print_step_info(prefix, global_step, info, result_summary, log_f):
  """Print all info at the current global step."""
  utils.print_out(
      "%sstep %d lr %g step-time %.2fs wps %.2fK ppl %.2f gN %.2f %s, %s" %
      (prefix, global_step, info["learning_rate"], info["avg_step_time"],
       info["speed"], info["train_ppl"], info["avg_grad_norm"], result_summary,
       time.ctime()),
      log_f)


def add_info_summaries(summary_writer, global_step, info):
  """Add stuffs in info to summaries."""
  excluded_list = ["learning_rate"]
  for key in info:
    if key not in excluded_list:
      utils.add_summary(summary_writer, global_step, key, info[key])

'''
使用stats更新

info:
  avg_step_time:每一步训练花费的时间
  avg_grad_norm:平均global_grad_norm是多大
  avg_sequence_count:平均一次处理多少个单词
  speed:平均一秒处理多少个单词(源+目标)
  train_ppl:训练的perplexity
  
steps_per_stats:更新stats经过的次数
返回:
    info["train_ppl"]是否溢出is_overflow
'''
def process_stats(stats, info, global_step, steps_per_stats, log_f):
  """Update info and check for overflow."""
  # Per-step info
  info["avg_step_time"] = stats["step_time"] / steps_per_stats
  info["avg_grad_norm"] = stats["grad_norm"] / steps_per_stats
  info["avg_sequence_count"] = stats["sequence_count"] / steps_per_stats
  info["speed"] = stats["word_count"] / (1000 * stats["step_time"])

  # Per-predict info
  info["train_ppl"] = (
      utils.safe_exp(stats["train_loss"] / stats["predict_count"]))

  # Check for overflow
  is_overflow = False
  train_ppl = info["train_ppl"]
  if math.isnan(train_ppl) or math.isinf(train_ppl) or train_ppl > 1e20:
    utils.print_out("  step %d overflow, stop early" % global_step,
                    log_f)
    is_overflow = True

  return is_overflow
'''

初始化tensor输入的initializer
返回:
stats:
info: 
start_train_time:时间戳
'''

def before_train(loaded_train_model, train_model, train_sess, global_step,
                 hparams, log_f):
  """Misc tasks to do before training."""
  stats = init_stats()
  info = {"train_ppl": 0.0, "speed": 0.0,
          "avg_step_time": 0.0,
          "avg_grad_norm": 0.0,
          "avg_sequence_count": 0.0,
          "learning_rate": loaded_train_model.learning_rate.eval(
              session=train_sess)}
  start_train_time = time.time()
  utils.print_out("# Start step %d, lr %g, %s" %
                  (global_step, info["learning_rate"], time.ctime()), log_f)

  # Initialize all of the iterators
  skip_count = hparams.batch_size * hparams.epoch_step
  utils.print_out("# Init train iterator, skipping %d elements" % skip_count)
  train_sess.run(
      train_model.iterator.initializer,
      feed_dict={train_model.skip_count_placeholder: skip_count})

  return stats, info, start_train_time


def get_model_creator(hparams):
  """Get the right model class depending on configuration."""
  if (hparams.encoder_type == "gnmt" or
      hparams.attention_architecture in ["gnmt", "gnmt_v2"]):
    model_creator = gnmt_model.GNMTModel
  elif hparams.attention_architecture == "standard":
    model_creator = attention_model.AttentionModel
  elif not hparams.attention:
    model_creator = nmt_model.Model
  else:
    raise ValueError("Unknown attention architecture %s" %
                     hparams.attention_architecture)
  return model_creator

'''

返回

final_eval_metrics:训练结束后各种
    metrics分数
    {metrics_name:metrics_score}
global_step:int,Train运行的步数
'''
def train(hparams, scope=None, target_session=""):
  """Train a translation model."""
  log_device_placement = hparams.log_device_placement
  out_dir = hparams.out_dir
  num_train_steps = hparams.num_train_steps
  steps_per_stats = hparams.steps_per_stats
  steps_per_external_eval = hparams.steps_per_external_eval
  steps_per_eval = 10 * steps_per_stats
  avg_ckpts = hparams.avg_ckpts

  if not steps_per_external_eval:
    steps_per_external_eval = 5 * steps_per_eval

  # Create model,获得了创建模型的构造
  model_creator = get_model_creator(hparams)
  train_model = model_helper.create_train_model(model_creator, hparams, scope)
  eval_model = model_helper.create_eval_model(model_creator, hparams, scope)
  infer_model = model_helper.create_infer_model(model_creator, hparams, scope)

  # Preload data for sample decoding.,测试文件源,目标文件
  dev_src_file = "%s.%s" % (hparams.dev_prefix, hparams.src)
  dev_tgt_file = "%s.%s" % (hparams.dev_prefix, hparams.tgt)
  sample_src_data = inference.load_data(dev_src_file)
  sample_tgt_data = inference.load_data(dev_tgt_file)

  summary_name = "train_log"
  model_dir = hparams.out_dir

  # Log and output files
  log_file = os.path.join(out_dir, "log_%d" % time.time())
  log_f = tf.gfile.GFile(log_file, mode="a")
  utils.print_out("# log_file=%s" % log_file, log_f)

  # TensorFlow model
  config_proto = utils.get_config_proto(
      log_device_placement=log_device_placement,
      num_intra_threads=hparams.num_intra_threads,
      num_inter_threads=hparams.num_inter_threads)
  train_sess = tf.Session(
      target=target_session, config=config_proto, graph=train_model.graph)
  eval_sess = tf.Session(
      target=target_session, config=config_proto, graph=eval_model.graph)
  infer_sess = tf.Session(
      target=target_session, config=config_proto, graph=infer_model.graph)
  #初始化(恢复参数)训练图
  with train_model.graph.as_default():
    loaded_train_model, global_step = model_helper.create_or_load_model(
        train_model.model, model_dir, train_sess, "train")

  # Summary writer
  summary_writer = tf.summary.FileWriter(
      os.path.join(out_dir, summary_name), train_model.graph)

  # First evaluation
  '''
  真实输入数据:sample_src_data,sample_tgt_data
  dev:使用eval_sess运行eval_model,模型参数在model_dir,
    需要的数据sample_src_data,和sample_tgt_data,计算eval_loss
  infer:使用infer_sess运行infer_model,模型参数在model_dir,
    只用数据和sample_tgt_data,
  '''
  run_full_eval(
      model_dir, infer_model, infer_sess,
      eval_model, eval_sess, hparams,
      summary_writer, sample_src_data,
      sample_tgt_data, avg_ckpts)

  last_stats_step = global_step
  last_eval_step = global_step
  last_external_eval_step = global_step

  # This is the training loop.
  stats, info, start_train_time = before_train(
      loaded_train_model, train_model, train_sess, global_step, hparams, log_f)
  '''
  训练概述:
    [t,t+steps_per_stats]区间做一次统计,记录在stat里面
    stat:记录了区间内产生的
        step_time:运行时长
        trainloss总计
        grad_global_norm总计
        predict_count:翻译的目标单词总计
        word_cout:预览的所有单词(源+目标)
    每steps_per_stats步后,汇总统计结果到info,结果输出到控制台
        avg_step_time:step_time/steps_per_stats
        avg_grad_norm:平均global_grad_norm是多大
        avg_sequence_count:平均一次处理多少个单词
        speed:平均一秒处理多少个单词(源+目标)
        train_ppl:训练的perplexity
        lr:
        如果train_ppl产生溢出,训练结束
        
        然后做
            1.保存模型参数到output_dir/translate.ckpt
            2.sample of src_data
            3.run_internal_eval
    每steps_per_eval后:
        1,保存模型到output_dir/translate.ckpt
        2.随机从sample_src翻译,并和sample_tgt对比
        3.run_internal_eval
    每steps_per_external_eval:后
            1.保存模型参数到output_dir/translate.ckpt
            2.sample of src_data
            3.run_external_eval
  '''
  while global_step < num_train_steps:
    ### Run a step ###
    start_time = time.time()
    try:
      step_result = loaded_train_model.train(train_sess)
      hparams.epoch_step += 1
    except tf.errors.OutOfRangeError:
      # Finished going through the training dataset.  Go to next epoch.
      hparams.epoch_step = 0
      utils.print_out(
          "# Finished an epoch, step %d. Perform external evaluation" %
          global_step)
      run_sample_decode(infer_model, infer_sess, model_dir, hparams,
                        summary_writer, sample_src_data, sample_tgt_data)
      run_external_eval(infer_model, infer_sess, model_dir, hparams,
                        summary_writer)

      if avg_ckpts:
        run_avg_external_eval(infer_model, infer_sess, model_dir, hparams,
                              summary_writer, global_step)

      train_sess.run(
          train_model.iterator.initializer,
          feed_dict={train_model.skip_count_placeholder: 0})
      continue
    '''
    step_result 本批次:
        train_loss:batch_size条记录的平均loss
        grad_norm:
        global_step
        word_count:当前batch个数据 源单词数+目标单词数
        batch_size:输入的batchsize
        learning_rate
        predict_count:当前batch中,目标单词数
    '''
    # Process step_result, accumulate stats, and write summary
    global_step, info["learning_rate"], step_summary = update_stats(
        stats, start_time, step_result)
    summary_writer.add_summary(step_summary, global_step)

    # Once in a while, we print statistics.
    if global_step - last_stats_step >= steps_per_stats:
      last_stats_step = global_step
      is_overflow = process_stats(
          stats, info, global_step, steps_per_stats, log_f)
      print_step_info("  ", global_step, info, get_best_results(hparams),
                      log_f)
      if is_overflow:
        break

      # Reset statistics
      stats = init_stats()

    if global_step - last_eval_step >= steps_per_eval:
      last_eval_step = global_step
      utils.print_out("# Save eval, global step %d" % global_step)
      add_info_summaries(summary_writer, global_step, info)

      # Save checkpoint
      loaded_train_model.saver.save(
          train_sess,
          os.path.join(out_dir, "translate.ckpt"),
          global_step=global_step)

      # Evaluate on dev/test
      run_sample_decode(infer_model, infer_sess,
                        model_dir, hparams, summary_writer, sample_src_data,
                        sample_tgt_data)
      run_internal_eval(
          eval_model, eval_sess, model_dir, hparams, summary_writer)

    if global_step - last_external_eval_step >= steps_per_external_eval:
      last_external_eval_step = global_step

      # Save checkpoint
      loaded_train_model.saver.save(
          train_sess,
          os.path.join(out_dir, "translate.ckpt"),
          global_step=global_step)
      run_sample_decode(infer_model, infer_sess,
                        model_dir, hparams, summary_writer, sample_src_data,
                        sample_tgt_data)
      run_external_eval(
          infer_model, infer_sess, model_dir,
          hparams, summary_writer)

      if avg_ckpts:
        run_avg_external_eval(infer_model, infer_sess, model_dir, hparams,
                              summary_writer, global_step)

  # Done training
  loaded_train_model.saver.save(
      train_sess,
      os.path.join(out_dir, "translate.ckpt"),
      global_step=global_step)

  (result_summary, _, final_eval_metrics) = (
      run_full_eval(
          model_dir, infer_model, infer_sess, eval_model, eval_sess, hparams,
          summary_writer, sample_src_data, sample_tgt_data, avg_ckpts))
  print_step_info("# Final, ", global_step, info, result_summary, log_f)
  utils.print_time("# Done training!", start_train_time)

  summary_writer.close()

  utils.print_out("# Start evaluating saved best models.")
  for metric in hparams.metrics:
    best_model_dir = getattr(hparams, "best_" + metric + "_dir")
    summary_writer = tf.summary.FileWriter(
        os.path.join(best_model_dir, summary_name), infer_model.graph)
    result_summary, best_global_step, _ = run_full_eval(
        best_model_dir, infer_model, infer_sess, eval_model, eval_sess, hparams,
        summary_writer, sample_src_data, sample_tgt_data)
    print_step_info("# Best %s, " % metric, best_global_step, info,
                    result_summary, log_f)
    summary_writer.close()

    if avg_ckpts:
      best_model_dir = getattr(hparams, "avg_best_" + metric + "_dir")
      summary_writer = tf.summary.FileWriter(
          os.path.join(best_model_dir, summary_name), infer_model.graph)
      result_summary, best_global_step, _ = run_full_eval(
          best_model_dir, infer_model, infer_sess, eval_model, eval_sess,
          hparams, summary_writer, sample_src_data, sample_tgt_data)
      print_step_info("# Averaged Best %s, " % metric, best_global_step, info,
                      result_summary, log_f)
      summary_writer.close()

  return final_eval_metrics, global_step

'''
输出类似:dev ppl 22,dev blue 33,dev accuracy 0.1的string
'''
def _format_results(name, ppl, scores, metrics):
  """Format results."""
  result_str = ""
  if ppl:
    result_str = "%s ppl %.2f" % (name, ppl)
  if scores:
    for metric in metrics:
      if result_str:
        result_str += ", %s %s %.1f" % (name, metric, scores[metric])
      else:
        result_str = "%s %s %.1f" % (name, metric, scores[metric])
  return result_str

'''
输出最好的metrics成绩,返回blue 2,accuracy .1
备注:在run_external_eval的时候,已经把最好的
分数保存在了hparams的best_..属性里面了
'''
def get_best_results(hparams):
  """Summary of the current best results."""
  tokens = []
  for metric in hparams.metrics:
    tokens.append("%s %.2f" % (metric, getattr(hparams, "best_" + metric)))
  return ", ".join(tokens)

  '''
  model:Evaluation 模型的抽象计算图
  sess:
  iterator:Evaluation 模型计算图的抽象输入
  iterator_feed_dict:计算图的实际输入
  global_step
  label:'dev'
  summary_writer:日志文件
  
  这里使用的数据源是超参数给出的dev_prefix
  
  本方法用构建好的网络(model,iterator)
    计算(sess)给定数据源(iterator_feed_dict)的perplexity,并写入日志(summary_writer,label,global_step)
  
  返回对dev_prefix开发集的perplexity,并在日志中
    记录dev_ppl:ppl_value
  '''
def _internal_eval(model, global_step, sess, iterator, iterator_feed_dict,
                   summary_writer, label):
  """Computing perplexity."""
  sess.run(iterator.initializer, feed_dict=iterator_feed_dict)
  ppl = model_helper.compute_perplexity(model, sess, label)
  utils.add_summary(summary_writer, global_step, "%s_ppl" % label, ppl)
  return ppl

'''
model:
    model_helper.InferModel.model属性,抽象的一个计算图谱
iterator:
    model_helper.InferModel.iterator,计算图片的抽象输入
iterator_src_placeholder:tensor(string[]),用于驱动iterator,告诉iterator输入源是哪些文件
iterator_batch_size_placeholder:tensor(int32),用于驱动iterator,告诉iterator batch多大

eval数据部分:
src_data:list(),char类型,用于eval|infer的真实源语言数据
tgt_data:list(),char类型,用于eval的真实目标语言数据
global_step:

执行一次翻译过程:
    打印:src_sentence
        :target_sentenct
        :nmt(src_sentence)
'''
def _sample_decode(model, global_step, sess, hparams, iterator, src_data,
                   tgt_data, iterator_src_placeholder,
                   iterator_batch_size_placeholder, summary_writer):
  """Pick a sentence and decode."""
  decode_id = random.randint(0, len(src_data) - 1)
  utils.print_out("  # %d" % decode_id)

 #准备好输入
    #[My name is zhangxk]--->tensor(string[])
  iterator_feed_dict = {
      iterator_src_placeholder: [src_data[decode_id]],
      iterator_batch_size_placeholder: 1,
  }
  #问题?为什么运行iterator.initializer要feed iterator_feed_dict
  sess.run(iterator.initializer, feed_dict=iterator_feed_dict)
  #翻译:nmt_output:[bw,N,?] or[N,?]
  nmt_outputs, attention_summary = model.decode(sess)

  if hparams.infer_mode == "beam_search":
    # get the top translation.
    nmt_outputs = nmt_outputs[0]

  translation = nmt_utils.get_translation(
      nmt_outputs,
      sent_id=0,
      tgt_eos=hparams.eos,
      subword_option=hparams.subword_option)
  utils.print_out("    src: %s" % src_data[decode_id])
  utils.print_out("    ref: %s" % tgt_data[decode_id])
  utils.print_out(b"    nmt: " + translation)

  # Summary
  if attention_summary is not None:
    summary_writer.add_summary(attention_summary, global_step)

'''
model:infer_model

infer_sess,
hparams
iterator:
iterator_feed_dict:feed_dict,已经feed好了src_placeholder,batch_placeholder
global_step(int),summary_writer:保存日志使用

tgt_file:翻译目标文件,计算blue的参考文件
label:'dev'

表述:
    使用sess,运行model,把输入iterator_feed_dict翻译结果输出
    到hparams.outdir/output_+label/transfile下面.
    根据hparams.metrics,与tgt_file对比得出scores,
    总结每一global_step的scores,输出到summary_writer日志中.
    
    
save_on_best:True的时候,到某个metrics在测试集合结果最好的时候,
    保存到best_+{best_metric_label}+_dir下面translate.ckpt
avg_ckpts:True:best_metric_label=avg_best_+metric
        False:best_metric_label=best_+metric
        
返回:scores:
    keys:metric_name
    value:metrics_score
'''
def _external_eval(model, global_step, sess, hparams, iterator,
                   iterator_feed_dict, tgt_file, label, summary_writer,
                   save_on_best, avg_ckpts=False):
  """External evaluation such as BLEU and ROUGE scores."""
  out_dir = hparams.out_dir
  decode = global_step > 0

  if avg_ckpts:
    label = "avg_" + label

  if decode:
    utils.print_out("# External evaluation, global step %d" % global_step)
    #输入数据准备完毕
  sess.run(iterator.initializer, feed_dict=iterator_feed_dict)

  output = os.path.join(out_dir, "output_%s" % label)
  #output:outputdir/output_dev
  scores = nmt_utils.decode_and_evaluate(
      label,
      model,
      sess,
      output,
      ref_file=tgt_file,
      metrics=hparams.metrics,
      subword_option=hparams.subword_option,
      beam_width=hparams.beam_width,
      tgt_eos=hparams.eos,
      decode=decode,
      infer_mode=hparams.infer_mode)
  # Save on best metrics
  if decode:
    for metric in hparams.metrics:
      if avg_ckpts:
        best_metric_label = "avg_best_" + metric
      else:
        best_metric_label = "best_" + metric

      utils.add_summary(summary_writer, global_step, "%s_%s" % (label, metric),
                        scores[metric])
      # metric: larger is better
      if save_on_best and scores[metric] > getattr(hparams, best_metric_label):
        setattr(hparams, best_metric_label, scores[metric])
        model.saver.save(
            sess,
            os.path.join(
                getattr(hparams, best_metric_label + "_dir"), "translate.ckpt"),
            global_step=model.global_step)
    utils.save_hparams(out_dir, hparams)
  return scores
