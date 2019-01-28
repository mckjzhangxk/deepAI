import numpy as np
import numpy.random as npr
import tensorflow as tf
from train_model.losses import classLosses,boxesLoss,landmarkLoss,calAccuracy

def tst_landmarkLoss():
    print('Test landmark loss!!')
    N = 384
    landmark=npr.rand(N,10)
    gt_landmark = npr.rand(N, 10)

    label = npr.randint(-2, 2, N)


    loss = 0
    cnt = 0
    for idx, v in enumerate(label):
        if v == -2:
            diff=gt_landmark[idx]-landmark[idx]
            loss+=diff.dot(diff)
            cnt+=1
    print(loss/cnt)

    LABEL = tf.placeholder(tf.float32, [N])
    GT_LANDMARK = tf.placeholder(tf.float32, [N, 10])
    LANDMARK = tf.placeholder(tf.float32, [N, 10])
    LOSS = landmarkLoss(LANDMARK,GT_LANDMARK,LABEL)
    with tf.Session() as sess:
        _cls = sess.run(LOSS, feed_dict={LABEL: label, LANDMARK: landmark,GT_LANDMARK:gt_landmark})
        print(_cls)

def tst_boxLoss():
    print('Test box loss!!!')
    N=384
    x1,y1=npr.rand(N),npr.rand(N)
    x2,y2=x1+npr.rand(N),y1+npr.rand(N)
    regbox=np.stack([x1,y1,x2,y2],axis=1)

    gt_x1, gt_y1 = npr.rand(N), npr.rand(N)
    gt_x2, gt_y2 = gt_x1 + npr.rand(N), gt_y1 + npr.rand(N)
    roi=np.stack([gt_x1,gt_y1,gt_x2,gt_y2],axis=1)


    label = npr.randint(-2, 2, N)
    loss = 0
    cnt = 0
    for idx, v in enumerate(label):
        if np.abs(v) == 1:
            loss+=(gt_x2[idx]-x2[idx])**2+(gt_x1[idx]-x1[idx])**2+(gt_y2[idx]-y2[idx])**2+(gt_y1[idx]-y1[idx])**2
            cnt+=1
    print(loss/cnt)


    LABEL = tf.placeholder(tf.float32, [N])
    RGEBOX = tf.placeholder(tf.float32, [N, 4])
    ROI = tf.placeholder(tf.float32, [N, 4])
    bs_loss = boxesLoss(RGEBOX,ROI,LABEL)
    with tf.Session() as sess:
        _cls = sess.run(bs_loss, feed_dict={LABEL: label, ROI: roi,RGEBOX:regbox})
        print(_cls)

def tst_ClassLoss():
    print('Test classification loss!!!')
    N = 380
    prob = np.random.rand(N)
    score = np.stack([1 - prob, prob], axis=1)

    label = npr.randint(-2, 2, N)

    loss = 0
    cnt = 0
    for idx, v in enumerate(label):
        if v == 1 or v == 0:
            loss += -np.log(score[idx][v])
            cnt += 1
    if cnt > 0:
        print(loss / cnt)
    else:
        print('nan')
    LABEL = tf.placeholder(tf.float32, [N])
    PROB = tf.placeholder(tf.float32, [N, 2])
    cls_loss = classLosses(PROB, LABEL)
    with tf.Session() as sess:
        _cls = sess.run(cls_loss, feed_dict={LABEL: label, PROB: score})
        print(_cls)


def tst_accuracy():
    print('Test accuracy!!!')
    N = 384
    prob = np.random.rand(N)
    score = np.stack([1 - prob, prob], axis=1)

    label = npr.randint(-2, 2, N)


    acc = 0
    cnt=0
    for idx, v in enumerate(label):
        if v == 1 or v == 0:
            logit=1 if prob[idx]>=0.5 else 0
            if logit==v:
                acc+=1
            cnt += 1
    if cnt > 0:
        print(acc / cnt)
    else:
        print('nan')
    LABEL = tf.placeholder(tf.float32, [N])
    PROB = tf.placeholder(tf.float32, [N, 2])
    ACC = calAccuracy(PROB, LABEL)
    with tf.Session() as sess:
        _cls = sess.run(ACC, feed_dict={LABEL: label, PROB: score})
        print(_cls)
if __name__ == '__main__':
    # tst_ClassLoss()
    # tst_boxLoss()
    # tst_landmarkLoss()
    tst_accuracy()