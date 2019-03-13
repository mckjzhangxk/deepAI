import tensorflow as tf
from train.trainNetwork import run_train

hparam = tf.contrib.training.HParams(
    mode='run_train',
    rnn_type='lstm',
    ndims=128,
    num_layers=2,
    num_output=2,
    batch_size=128,
    dropout=0.0,
    forget_bias=1.0,
    residual=True,
    perodic=30,
    Tmax=300,  # 序列的最大长度,是文件的宽度/特征数量
    lr=1e-4,
    solver='adam',
    num_train_steps=95000,
    decay_scheme='luong5',  # "luong5", "luong10", "luong234"
    max_gradient_norm=5,
    features=6,

    # train_datafile='/home/zhangxk/AIProject/ippack/vpndata/run_train.txt',
    # eval_datafile='/home/zhangxk/AIProject/ippack/vpndata/run_train.txt',
    train_datafile='../data/resource/data',
    eval_datafile='../data/resource/data',

    log_dir='log',
    model_dir='models/MyNet',
    # model_dir='best/best_acc',
    steps_per_state=10,
    max_keeps=5,
    scope='VPNNetWork',
    best_model_path='best',

    soft_placement=True,
    log_device=False
)
run_train(hparam)