from recognition_facenet.detect_face import create_mtcnn
import tensorflow as tf
import numpy as np


sess=tf.Session()
pnet,rnet,onet=create_mtcnn(sess,'/home/zhangxk/mysite/facenet/align')



def classify_loss(label,logit,mask):
    logit=tf.squeeze(logit)
    loss=tf.nn.softmax_cross_entropy_with_logits_v2(labels=label,logits=logit)
    loss=tf.reduce_mean(loss*mask)
    return loss
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

def preparePnetLoss(Y,MASK):
    def pnet_loss(Label, Logit, Ybox, YhatBox, mask):
        ls1 = classify_loss(Label, Logit, mask[:, 0])
        ls2 = box_loss(Ybox, YhatBox, mask[:, 1])
        return ls1 + 0.5 * ls2

    g = tf.get_default_graph()

    #输入
    X = g.get_tensor_by_name("pnet/input:0")
    Y_BOX = tf.placeholder(dtype=tf.float32, shape=[None,4])

    #输出
    YHAT = g.get_tensor_by_name("pnet/prob1:0")
    YHAT_BOX = g.get_tensor_by_name("pnet/conv4-2/BiasAdd:0")

    loss=pnet_loss(Y,YHAT,Y_BOX,YHAT_BOX,MASK)
    return X,Y,Y_BOX,loss

def prepareRnetLoss(Y,MASK):
    def rnet_loss(Label, Logit, Ybox, YhatBox, mask):
        ls1 = classify_loss(Label, Logit, mask[:, 0])
        ls2 = box_loss(Ybox, YhatBox, mask[:, 1])
        return ls1 + 0.5 * ls2

    g = tf.get_default_graph()
    #输入
    X = g.get_tensor_by_name("rnet/input:0")
    Y_BOX = tf.placeholder(dtype=tf.float32, shape=[None, 4])

    #输出
    YHAT = g.get_tensor_by_name("rnet/prob1:0")
    YHAT_BOX = g.get_tensor_by_name("rnet/conv5-2/conv5-2:0")

    loss=rnet_loss(Y,YHAT,Y_BOX,YHAT_BOX,MASK)
    return X,Y,Y_BOX,loss

def prepareOnetLoss(Y,MASK):
    def onet_loss(Label, Logit, Ybox, YhatBox, mask):
        ls1 = classify_loss(Label, Logit, mask[:, 0])
        ls2 = box_loss(Ybox, YhatBox, mask[:, 1])
        return ls1 + 0.5 * ls2

    g = tf.get_default_graph()
    #输入
    X = g.get_tensor_by_name("onet/input:0")
    Y_BOX = tf.placeholder(dtype=tf.float32, shape=[None, 4])

    #输出
    YHAT = g.get_tensor_by_name("onet/prob1:0")
    YHAT_BOX = g.get_tensor_by_name("onet/conv6-2/conv6-2:0")

    loss=onet_loss(Y,YHAT,Y_BOX,YHAT_BOX,MASK)
    return X,Y,Y_BOX,loss


def prepareLossAndInput():
    Y = tf.placeholder(dtype=tf.float32, shape=[None, 2])

    haveObject=Y[:,1]
    one=tf.ones_like(Y[:,1])
    MASK=tf.stack([one,haveObject],axis=1)

    pX,_, pYBOX,pLOSS=preparePnetLoss(Y,MASK)
    rX,_, rYBOX,rLOSS=prepareRnetLoss(Y,MASK)
    oX,_, oYBOX,oLOSS=prepareOnetLoss(Y,MASK)
    LOSS=pLOSS+rLOSS+oLOSS

    return [pX,rX,oX,pYBOX,rYBOX,oYBOX,Y,LOSS]

pX,rX,oX,pYBOX,rYBOX,oYBOX,Y,LOSS=prepareLossAndInput()
optimizer=tf.train.AdamOptimizer().minimize(LOSS)

#####################################################################
#####################################################################
#####################################################################
N=32
px=np.random.rand(N,12,12,3)
rx=np.random.rand(N,24,24,3)
ox=np.random.rand(N,48,48,3)

pbox=np.random.rand(N,4)
rbox=np.random.rand(N,4)
obox=np.random.rand(N,4)

y=np.random.randint(0,2,(N,2))
#####################################################################
#####################################################################
#####################################################################

sess.run(tf.global_variables_initializer())
for i in range(10):
    _loss,_=sess.run([LOSS,optimizer], feed_dict={
        pX:px     ,rX:rx     ,oX:ox,
        pYBOX:pbox,rYBOX:rbox,oYBOX:obox,
        Y:y
    })
    print(_loss)
