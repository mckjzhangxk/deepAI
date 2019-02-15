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

"""Utility functions specifically for NMT."""
from __future__ import print_function

import codecs
import time
import numpy as np
import tensorflow as tf

from ..utils import evaluation_utils
from ..utils import misc_utils as utils

__all__ = ["decode_and_evaluate", "get_translation"]
'''
使用sess,运行model(InferModel),获得翻译的结果,把结果输出到trans_file中,
每一句话翻译num_translations_per_input遍(对infer_mode有用,greed没有,beam还会参考beam_width).

最后把trans_file和ref_file做比较,得出结果
返回:
evaluation_scores:dict
  key:(bleu,rouge,accuracy)
  value:score
'''

def decode_and_evaluate(name,
                        model,
                        sess,
                        trans_file,
                        ref_file,
                        metrics,
                        subword_option,
                        beam_width,
                        tgt_eos,
                        num_translations_per_input=1,
                        decode=True,
                        infer_mode="greedy"):
  """Decode a test set and compute a score according to the evaluation task."""
  # Decode
  if decode:
    utils.print_out("  decoding to output %s" % trans_file)

    start_time = time.time()
    num_sentences = 0
    with codecs.getwriter("utf-8")(
        tf.gfile.GFile(trans_file, mode="wb")) as trans_f:
      trans_f.write("")  # Write empty string to ensure file is created.

      if infer_mode == "greedy":
        num_translations_per_input = 1
      elif infer_mode == "beam_search":
        num_translations_per_input = min(num_translations_per_input, beam_width)
      #把翻译的结果全部输出到了trans_file文件里面,num_sentences是多少个翻译的源句子,
      # 每个源句子有bw个备份
      while True:
        try:
          #nmt_outputs:(N,T) or(bw,N,T),得到了
          nmt_outputs, _ = model.decode(sess)
          if infer_mode != "beam_search":
            nmt_outputs = np.expand_dims(nmt_outputs, 0)
          # nmt_outputs:(1,N,T) or(bw,N,T)
          batch_size = nmt_outputs.shape[1]
          num_sentences += batch_size

          #nmt_outputs是(bw,N,T)的数组,也就是说要把数组转成string bw*n次
          # 每一句话有bw个翻译版本
          for sent_id in range(batch_size):
            for beam_id in range(num_translations_per_input):
              translation = get_translation(
                  nmt_outputs[beam_id],
                  sent_id,
                  tgt_eos=tgt_eos,
                  subword_option=subword_option)
              trans_f.write((translation + b"\n").decode("utf-8"))
        except tf.errors.OutOfRangeError:
          utils.print_time(
              "  done, num sentences %d, num translations per input %d" %
              (num_sentences, num_translations_per_input), start_time)
          break

  # Evaluation
  evaluation_scores = {}
  if ref_file and tf.gfile.Exists(trans_file):
    for metric in metrics:
      score = evaluation_utils.evaluate(
          ref_file,
          trans_file,
          metric,
          subword_option=subword_option)
      evaluation_scores[metric] = score
      utils.print_out("  %s %s: %.1f" % (metric, name, score))

  return evaluation_scores

'''
nmt_outputs:(N,?) nmt的输出
sent_id:选择输出的句子索引

nmt_outputs[sent_id]--->str(nmt_outputs[sent_id])
输出翻译完成的字符串!!
'''
def get_translation(nmt_outputs, sent_id, tgt_eos, subword_option):
  """Given batch decoding outputs, select a sentence and turn to text."""
  if tgt_eos: tgt_eos = tgt_eos.encode("utf-8")
  # Select a sentence
  output = nmt_outputs[sent_id, :].tolist()

  # If there is an eos symbol in outputs, cut them at that point.
  if tgt_eos and tgt_eos in output:
    output = output[:output.index(tgt_eos)]
  #output.shape[T]
  if subword_option == "bpe":  # BPE
    translation = utils.format_bpe_text(output)
  elif subword_option == "spm":  # SPM
    translation = utils.format_spm_text(output)
  else:
    translation = utils.format_text(output)

  return translation
