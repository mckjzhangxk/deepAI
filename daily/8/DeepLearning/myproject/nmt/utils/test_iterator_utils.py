import tensorflow as tf
from utils.iterator_utils import get_iterator,get_infer_iterator
from modelHelper import createTrainModel
from utils.nmt_utils import get_translation

hparam=tf.contrib.training.HParams(
    train_src='../data/train.vi',
    train_tgt='../data/train.en',
    dev_src='../data/train.vi',
    dev_tgt='../data/train.en',
    test_src='',
    test_tgt='',
    vocab_src='../data/vocab.vi',
    vocab_tgt='../data/vocab.en',

    SOS='<sos>',
    EOS='<eos>',
    UNK='<unk>',
    batch_size=128,

    #####网络相关参数###########
    scope='nmt',
    encode_type='uni',
    rnn_type='lstm',
    emb_size=128,
    ndim=128,
    num_layer=2,
    activation_fn=None,
    dropout=0.0,
    forget_bias=1.0,
    residual_layer=False,
    share_vocab=False,

    infer_mode='beam_search',
    beam_width=3,
    ##########训练参数相关###########
    optimizer='adam',
    lr=1e-2,
    decay_scheme='luong5', #luong5,luong10
    warmup_steps=1000,
    warmup_scheme='t2t',
    max_norm=5,
    ##########训练流程相关###########
    num_train=6000,
    steps_per_stat=10,
    steps_per_innernal_eval=50,
    steps_per_external_eval=100,

    model_path='weights/model',
    max_to_keep=5,
    ckpt=False,
    checkpoint_path='weights/bestmodel',
    log_dir='log'
)

def mytest_iterator():
    src_dataset=tf.data.TextLineDataset(hparam.train_src)
    tgt_dataset=tf.data.TextLineDataset(hparam.train_tgt)
    src,tgt_in,tgt_out,src_seq_len,tgt_seq_len,initializer,_=get_iterator(src_dataset,tgt_dataset,hparam)

    with tf.Session() as sess:
        sess.run(tf.tables_initializer())
        sess.run(initializer)

        for i in range(1):
            try:
                _src,_tgt_in,_tgt_out,_src_seq_len,_tgt_seq_len=sess.run([src,tgt_in,tgt_out,src_seq_len,tgt_seq_len])
                print('src',_src)
                print('tgt_in', _tgt_in)
                print('tgt_out',_tgt_out)
                print('src_seq_len', _src_seq_len)
                print('tgt_seq_len', _tgt_seq_len)
            except tf.errors.OutOfRangeError:
                print('xxxxxxxxxxxxxxx')
                sess.run(initializer)
def mytest_infer_interator():
    src_dataset = tf.data.TextLineDataset(hparam.train_src)
    myinput= get_infer_iterator(src_dataset, hparam)
    ss=myinput.reverse_table.lookup(myinput.src)
    with tf.Session() as sess:
        sess.run(tf.tables_initializer())
        sess.run(myinput.initializer)
        for i in range(5):
            try:
                _src, _src_seq_len,cc= sess.run(
                    [myinput.src,myinput.src_seq_len]+[ss])
                print('src', _src)

                print('src_seq_len', _src_seq_len)
                print('reverce')
                for i,c in enumerate(cc):
                    print(get_translation(cc,i,hparam.EOS))
            except tf.errors.OutOfRangeError:
                print('xxxxxxxxxxxxxxx')
                sess.run(myinput.initializer)
# mytest_iterator()

mytest_infer_interator()