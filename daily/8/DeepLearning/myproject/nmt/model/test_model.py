import tensorflow as tf
from utils.iterator_utils import get_iterator
from model.BaseModel import Model

hparam=tf.contrib.training.HParams(
    train_src='../data/train.en',
    train_tgt='../data/train.vi',
    dev_src='',
    dev_tgt='',
    test_src='',
    test_tgt='',
    vocab_src='../data/vocab.en',
    vocab_tgt='../data/vocab.vi',

    SOS='<sos>',
    EOS='<eos>',
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
    modir_dir='',
    max_keep=5,
    checkpoint_path='',
    log_dir=''
)
src_dataset=tf.data.TextLineDataset(hparam.train_src)
tgt_dataset=tf.data.TextLineDataset(hparam.train_tgt)
batchinput=get_iterator(src_dataset,tgt_dataset,hparam)
model=Model(batchinput,'train',hparam)

for v in tf.trainable_variables():
    print(v)

sess=tf.Session()

sess.run(tf.tables_initializer())
sess.run(tf.global_variables_initializer())
sess.run(batchinput.initializer)

for i in range(hparam.num_train):
    try:
        # for training
        # _loss,_sample_id=sess.run([model._loss,model._sample_id])
        # print(_sample_id.shape)
        # print(_loss)
        #
        # _sample_id = sess.run(model._sample_id)
        # print(_sample_id.shape)
        # print(_sample_id)
        rs=model.train(sess)
        print('step %d,loss %f,global norm %.2f,after clip %.2f,# of words %d,# of predict %d'%(
                                                  rs.global_step,
                                                  rs.loss,
                                                  rs.global_norm,
                                                  rs.clip_global_norm,
                                                  rs.word_count,
                                                  rs.predict_count))
    except tf.errors.OutOfRangeError:
        print('xxxxxxxxxxxxxxxxxxxxx')
        sess.run(batchinput.initializer)
