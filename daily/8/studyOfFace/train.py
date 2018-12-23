from mylib.detect.detect_face import create_mtcnn,p_stage,r_stage,o_stage,drawLandMarks,drawDectectBox
from mylib.db.DataSource import WIDERDATA
import tensorflow as tf
import numpy as np
import os




def variable_summary(var,name='mysummary'):
    tf.summary.scalar(name,var)



def classify_loss(label, logit_prob, mask,eps=1e-7):
    label = tf.squeeze(label)
    logit_prob=tf.squeeze(logit_prob)

    loss=-tf.reduce_sum(label * tf.log(logit_prob+eps), 1)
    # loss=tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logit_prob)
    loss=tf.reduce_mean(loss*mask)
    return loss


def lrRate(lr, decay_steps, decay_rate, global_steps):
    ret = tf.train.exponential_decay(
        lr,
        global_steps,
        decay_steps,
        decay_rate,
        name='LearnRate'
    )
    variable_summary(ret,'LearnRate')
    return ret

def initial_variable(sess):
    uninit_vars = tf.report_uninitialized_variables()
    need_init_vars=sess.run(uninit_vars)

    op=[]
    for v in need_init_vars:
        name = v.decode("utf-8")
        init_op=[v.initializer for v in tf.global_variables(name)]
        op.extend(init_op)

    sess.run(op)


'''
Y,Y1表示box的左上角和右下角的坐标,shape=[?,1,1,4]
返回loss=mean(|Y-Y1|^2)
'''
def box_loss(Y,Y1,mask):
    Y=tf.squeeze(Y)
    Y1 = tf.squeeze(Y1)
    #Y ,shape[?,1,1,4]

    ret=tf.reduce_sum((Y-Y1)**2,axis=-1) #[?,1,1]
    # ret=tf.squeeze(ret) #[?,]
    ret=tf.reduce_mean(ret*mask)
    return ret

def preparePnetLoss(Y,Y_BOX,MASK):
    def pnet_loss(Label, Logit, Ybox, YhatBox, mask):
        with tf.name_scope('pnet'):
            with tf.name_scope('entropyLoss'):
                ls1 = classify_loss(Label, Logit, mask[:, 0])
                variable_summary(ls1)
            with tf.name_scope('regressorLoss'):
                ls2 = box_loss(Ybox, YhatBox, mask[:, 1])
                variable_summary(ls2)
            with tf.name_scope('totalLoss'):
                ls=ls1 + 0.5 * ls2
                variable_summary(ls)
        return ls

    g = tf.get_default_graph()

    #输入
    X = g.get_tensor_by_name("pnet/input:0")

    #输出
    YHAT = g.get_tensor_by_name("pnet/prob1:0")
    YHAT_BOX = g.get_tensor_by_name("pnet/conv4-2/BiasAdd:0")

    loss=pnet_loss(Y,YHAT,Y_BOX,YHAT_BOX,MASK)
    return X,Y,Y_BOX,loss

def prepareRnetLoss(Y,Y_BOX,MASK):
    def rnet_loss(Label, Logit, Ybox, YhatBox, mask):
        with tf.name_scope('rnet'):
            with tf.name_scope('entropyLoss'):
                ls1 = classify_loss(Label, Logit, mask[:, 0])
                variable_summary(ls1)
            with tf.name_scope('regressorLoss'):
                ls2 = box_loss(Ybox, YhatBox, mask[:, 1])
                variable_summary(ls2)
            with tf.name_scope('totalLoss'):
                ls=ls1 + 0.5 * ls2
                variable_summary(ls)
        return ls

    g = tf.get_default_graph()
    #输入
    X = g.get_tensor_by_name("rnet/input:0")

    #输出
    YHAT = g.get_tensor_by_name("rnet/prob1:0")
    YHAT_BOX = g.get_tensor_by_name("rnet/conv5-2/conv5-2:0")

    loss=rnet_loss(Y,YHAT,Y_BOX,YHAT_BOX,MASK)
    return X,Y,Y_BOX,loss

def prepareOnetLoss(Y,Y_BOX,MASK):
    def onet_loss(Label, Logit, Ybox, YhatBox, mask):
        with tf.name_scope('onet'):
            with tf.name_scope('entropyLoss'):
                ls1 = classify_loss(Label, Logit, mask[:, 0])
                variable_summary(ls1)
            with tf.name_scope('regressorLoss'):
                ls2 = box_loss(Ybox, YhatBox, mask[:, 1])
                variable_summary(ls2)
            with tf.name_scope('totalLoss'):
                ls=ls1 + 0.5 * ls2
                variable_summary(ls)
        return ls

    g = tf.get_default_graph()
    #输入
    X = g.get_tensor_by_name("onet/input:0")

    #输出
    YHAT = g.get_tensor_by_name("onet/prob1:0")
    YHAT_BOX = g.get_tensor_by_name("onet/conv6-2/conv6-2:0")

    loss=onet_loss(Y,YHAT,Y_BOX,YHAT_BOX,MASK)
    return X,Y,Y_BOX,loss


def prepareLossAndInput():
    Y = tf.placeholder(dtype=tf.float32, shape=[None, 2])
    YBOX = tf.placeholder(dtype=tf.float32, shape=[None, 4])

    haveObject=Y[:,1]
    one=tf.ones_like(Y[:,1])
    MASK=tf.stack([one,haveObject],axis=1)

    pX,_, _,pLOSS=preparePnetLoss(Y,YBOX,MASK)
    rX,_, _,rLOSS=prepareRnetLoss(Y,YBOX,MASK)
    oX,_, _,oLOSS=prepareOnetLoss(Y,YBOX,MASK)

    with tf.name_scope('AllLoss'):
        LOSS=pLOSS+rLOSS+oLOSS
        variable_summary(LOSS)
    return [pX,rX,oX,Y,YBOX,LOSS]
#train configuration
batchSize=128
epochs=100
logdir='/home/zxk/PycharmProjects/deepAI/daily/8/studyOfFace/logs'
modeldir=os.path.join(logdir,'models','facedect.ckpt')
path='/home/zxk/AI/data/widerface/WIDER_train/samples'
summaryPerSteps=50
lr=0.001
decay_rate=0.96
decay_steps=26100

#define network
sess=tf.Session()
pnet,rnet,onet=create_mtcnn(sess)

pX,rX,oX,Y,YBOX,LOSS=prepareLossAndInput()
global_steps = tf.Variable(0, trainable=False)
OPTIMIZER=tf.train.AdamOptimizer(lrRate(lr,decay_steps,decay_rate,global_steps)).minimize(LOSS,global_step=global_steps)
MERGED=tf.summary.merge_all()
SUMMARY_WRITER=tf.summary.FileWriter(logdir, sess.graph)
SAVER=tf.train.Saver(max_to_keep=1000)

#before run solver
source=WIDERDATA(path,True)
# sess.run(tf.global_variables_initializer())
initial_variable(sess)

loopsPerEpochs=source.numExamples//batchSize +1

for e in range(epochs):
    avg_loss=0
    SAVER.save(sess, modeldir, e)
    for step in range(loopsPerEpochs):
        px, rx, ox, y, ybox = source.next(128,True)
        px = (px - 127.5) * 0.0078125
        rx = (rx - 127.5) * 0.0078125
        ox = (ox - 127.5) * 0.0078125

        feed={
            pX: px, rX: rx, oX: ox,
            Y: y, YBOX: ybox
        }

        _loss, _ = sess.run([LOSS, OPTIMIZER], feed_dict=feed)
        avg_loss+= _loss / loopsPerEpochs
        if step%summaryPerSteps==0:
            merged=sess.run(MERGED, feed_dict=feed)

            SUMMARY_WRITER.add_summary(merged, step + e * loopsPerEpochs)
            SUMMARY_WRITER.flush()
    print('End of epochs %d,loss is %.3f' % (e, avg_loss))
SUMMARY_WRITER.close()
