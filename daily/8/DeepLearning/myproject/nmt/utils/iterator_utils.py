import tensorflow as tf
import collections

class BatchInput(collections.namedtuple('BatchInput',['src','tgt_inp','tgt_out','src_seq_len','tgt_seq_len','initializer','reverse_vocab'])):pass


def get_iterator(src_dataset,tgt_dataset,hparam):
    '''
    src_dataset:Dataset类型，对应hparam.train_src/dev_src文件
    tgt_dataset:Dataset类型，对应hparam.train_tgt/dev_tgt文件
    hparam:使用到的参数:
        vocab_src,vocab_tgt
        EOS,SOS
        batch_size
    把数据源进行如下处理：
        1.字符串转字符数组
        2.过滤掉空行
        3.查表，字符串转索引
        4.src,tgt->(src,tgt_inp,tgt_out)
        5.(src,tgt_inp,tgt_out)-->(src,tgt_inp,tgt_out,src_seq_len,tgt_seq_len)
        6.dataset->batch tensor
    
    返回:(src,tgt_in,tgt_out,src_seq_len,tgt_seq_len,datasource_initializer)
        src:(?,Ts,int64)
        tgt_inp:(?,Tout,int64)
        tgt_out:(?,Tout,int64)
        src_seq_len:(?,int32)
        tgt_seq_len:(?,int32)
        
        备注：
            1.?一般是batch_size,除了最后一行除外
            2.Ts:一批来自src_dataset中最长的序列数量
            3.Tout一批来自tgt_dataset中最长的序列数量+1
    :param src_dataset: 
    :param tgt_dataset: 
    :param hparam: 
    :return: 
    '''
    vocab_src=tf.contrib.lookup.index_table_from_file(hparam.vocab_src,default_value=0)
    vocab_tgt=tf.contrib.lookup.index_table_from_file(hparam.vocab_tgt, default_value=0)

    SOS,EOS,UNK=hparam.SOS,hparam.EOS,hparam.EOS

    src_eos_id=vocab_src.lookup(tf.constant([EOS]))
    tgt_sos_id=vocab_tgt.lookup(tf.constant([SOS]))
    tgt_eos_id=vocab_tgt.lookup(tf.constant([EOS]))


    dataset=tf.data.Dataset.zip((src_dataset,tgt_dataset))
    dataset=dataset.map(lambda src,tgt:
                        (tf.string_split([src]).values,tf.string_split([tgt]).values)
                        )
    dataset=dataset.filter(lambda src,tgt:
                           tf.logical_and(
                               tf.size(src)>0,tf.size(tgt)>0
                           ))
    dataset=dataset.map(lambda src,tgt:
                        (vocab_src.lookup(src),vocab_tgt.lookup(tgt))
                        )

    dataset=dataset.map(lambda src,tgt:
                        (src,
                         tf.concat( [tgt_sos_id,tgt] ,0),
                         tf.concat( [tgt,tgt_eos_id] ,0)
                         )
                        )
    dataset=dataset.map(lambda src,tgt_in,tgt_out:
                        (src,tgt_in,tgt_out,tf.size(src),tf.size(tgt_in))
                        )

    dataset=dataset.padded_batch(batch_size=hparam.batch_size,
                         padded_shapes=(
                             tf.TensorShape([None]),
                             tf.TensorShape([None]),
                             tf.TensorShape([None]),
                             tf.TensorShape([]),
                             tf.TensorShape([])
                         ),
                         padding_values=(src_eos_id[0],tgt_eos_id[0],tgt_eos_id[0],0,0)
                         )
    batch_iter=dataset.make_initializable_iterator()

    src,tgt_inp,tgt_out,src_seq_len,tgt_seq_len=batch_iter.get_next()

    return BatchInput(src,
                      tgt_inp,
                      tgt_out,
                      src_seq_len,
                      tgt_seq_len,
                      batch_iter.initializer,
                      None)


def get_infer_iterator(src_dataset,hparam):
    '''
    为inference 设置迭代器，与get_iterator不同的是，返回对象不需要tgt,
    但多了reverse_vocab
    
    返回:(src,src_seq_len,datasource_initializer)
        src:(?,Ts,int64)
        src_seq_len:(?,int32)
        备注：
            1.?一般是batch_size,除了最后一行除外
            2.Ts:一批来自src_dataset中最长的序列数量

    :param src_dataset: 
    :param hparam: 
    :return: 
    '''
    SOS, EOS, UNK = hparam.SOS, hparam.EOS, hparam.EOS
    vocab_src=tf.contrib.lookup.index_table_from_file(hparam.vocab_src,default_value=0)
    reverse_vocab_tgt=tf.contrib.lookup.index_to_string_table_from_file(hparam.vocab_tgt, default_value=UNK)


    src_eos_id=vocab_src.lookup(tf.constant([EOS]))


    dataset=src_dataset
    dataset=dataset.map(lambda src:(tf.string_split([src]).values))
    dataset=dataset.map(lambda src:vocab_src.lookup(src))
    dataset=dataset.map(lambda src:(src,tf.size(src)))

    dataset=dataset.padded_batch(batch_size=hparam.batch_size,
                         padded_shapes=(
                             tf.TensorShape([None]),
                             tf.TensorShape([]),
                         ),
                         padding_values=(src_eos_id[0],0)
                         )
    batch_iter=dataset.make_initializable_iterator()

    src,src_seq_len=batch_iter.get_next()

    return BatchInput(src,
                      None,
                      None,
                      src_seq_len,
                      None,
                      batch_iter.initializer,
                      reverse_vocab_tgt)