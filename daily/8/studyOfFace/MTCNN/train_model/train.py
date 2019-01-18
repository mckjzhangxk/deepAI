from Configure import PNET_DATASET_PATH
from  utils.tf_utils import readTFRecord
from model.mtcnn_model import createPNet
from train_model.losses import classLosses,boxesLoss,calAccuracy
import tensorflow as tf
import os


svConf=None
def prepare():
    if not os.path.exists(svConf.MODEL_LOG_DIR):
        os.mkdir(svConf.MODEL_LOG_DIR)
    if not os.path.exists(svConf.MODEL_CHECKPOINT_DIR):
        os.mkdir(svConf.MODEL_CHECKPOINT_DIR)

def getInput(image_size):
    tf_filename=os.path.join(PNET_DATASET_PATH,'PNet_shuffle')
    assert os.path.exists(tf_filename) ,'PNet TFRecord does not exist'
    image_batch,label_batch,roi_batch=readTFRecord(tf_filename,svConf.BATCH_SIZE,image_size)
    return image_batch,label_batch,roi_batch

def buildModel(input_images):
    p_prob, p_regbox = createPNet(input_images, trainable=True)
    return p_prob,p_regbox

def buildLoss(prob,regbox,label,roi):
    cls_loss=classLosses(prob,label)
    reg_loss=boxesLoss(regbox,roi,label)
    total_loss=cls_loss+0.5*reg_loss
    acc=calAccuracy(prob,label)

    tf.summary.scalar('cls_loss',cls_loss)
    tf.summary.scalar('reg_loss',reg_loss)
    tf.summary.scalar('total_loss', total_loss)
    tf.summary.scalar('accuracy', acc)

    return total_loss,acc

def buildTarget(loss):
    LoopPerEpoch=svConf.EXAMPLES//svConf.BATCH_SIZE+1

    global_step=tf.Variable(0,trainable=False)
    lr=tf.train.exponential_decay(svConf.LR,
                                  global_step,
                                  decay_steps=LoopPerEpoch,
                                  decay_rate=svConf.DECAY_FACTOR,
                                  staircase=True)
    tf.summary.scalar('learn rate',lr)
    optimizer=tf.train.AdamOptimizer(lr).minimize(loss,global_step)
    return optimizer

def start_train(model_name):
    prepare()
    # 第一步,获取输入
    image_batch, label_batch, roi_batch = getInput(svConf.IMG_SIZE)
    # 第二部,搭建网络
    p_prob, p_regbox = buildModel(image_batch)
    # 第三部,获得loss, class_loss,reg_loss,l2_loss,以及accuracy

    dis_total_loss, dis_acc = buildLoss(p_prob, p_regbox, label_batch, roi_batch)
    # 第四步,定义训练算法,每次训练
    op_optimizer = buildTarget(dis_total_loss)
    op_summary = tf.summary.merge_all()

    # 开始训练

    MAX_STEPS = svConf.EPOCH * (svConf.EXAMPLES // svConf.LoopPerEpoch)

    with tf.Session() as sess:
        saver = tf.train.Saver(max_to_keep=0)
        # begin
        coord = tf.train.Coordinator()
        # begin enqueue thread
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        with tf.summary.FileWriter(svConf.MODEL_LOG_DIR, sess.graph) as writer:
            try:
                sess.run(tf.global_variables_initializer())
                for i in range(MAX_STEPS):
                    sess.run([op_optimizer])
                    if i % svConf.DISPLAY_EVERY == 0:
                        _acc, _loss, _summary = sess.run([dis_acc, dis_total_loss, op_summary])
                        print('Total Loss is %.3f,Accuracy is %.3f' % (_loss, _acc))
                    if i % svConf.LoopPerEpoch == 0:
                        saver.save(sess, os.path.join(svConf.MODEL_CHECKPOINT_DIR,model_name),i // svConf.LoopPerEpoch)
            finally:
                coord.request_stop()

            coord.join(threads)
