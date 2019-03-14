import codecs
import os
import tensorflow as tf


def load_vocab(vocab_file):
    '''
    从vocab_file加载单词:
  返回:
    vocab:[word1,word2....]
    vocab_size:单词数量
    :param vocab_file: 
    :return: 
    '''
    vocab = []
    with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
        vocab_size = 0
        for word in f:
          vocab_size += 1
          vocab.append(word.strip())
    return vocab, vocab_size

def check_vocab(vocab_file):
    '''
    检查单词文件vocab_file,是否存在,不存在抛异常
    返回单词数量,单词文件路径
    :param vocab_file: 
    :param out_dir: 
    :param check_special_token: 
    :param sos: 
    :param eos: 
    :param unk: 
    :return: 
    '''

    if tf.gfile.Exists(vocab_file):
        vocab, vocab_size = load_vocab(vocab_file)
    else:
        raise ValueError("vocab_file '%s' does not exist." % vocab_file)
    vocab_size = len(vocab)
    return vocab_size, vocab_file



def get_special_word_id(hparam):
    tb = tf.contrib.lookup.index_table_from_file(hparam.vocab_tgt)
    tgt_sos_id=tb.lookup(tf.constant([hparam.SOS]))[0]
    tgt_eos_id=tb.lookup(tf.constant([hparam.EOS]))[0]

    return tf.to_int32(tgt_sos_id),tf.to_int32(tgt_eos_id)
