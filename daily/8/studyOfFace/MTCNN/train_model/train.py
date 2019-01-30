import tensorflow as tf
import os
import numpy as np

getInput=None
buildModel=None
buildLoss=None
validateInput=None
validModel=None
validateAccuracy=None
validate=False

svConf=None

def prepare():
    if not os.path.exists(svConf.MODEL_LOG_DIR):
        os.mkdir(svConf.MODEL_LOG_DIR)
    if not os.path.exists(svConf.MODEL_CHECKPOINT_DIR):
        os.mkdir(svConf.MODEL_CHECKPOINT_DIR)

def image_color_distort(inputs):
    inputs = tf.image.random_contrast(inputs, lower=0.5, upper=1.5)
    inputs = tf.image.random_brightness(inputs, max_delta=0.2)
    inputs = tf.image.random_hue(inputs,max_delta= 0.2)
    inputs = tf.image.random_saturation(inputs,lower = 0.5, upper= 1.5)
    return inputs

def buildTarget(loss):
    LoopPerEpoch=svConf.EXAMPLES//svConf.BATCH_SIZE+1

    global_step=tf.Variable(0,trainable=False)

    boundary=[epoch*LoopPerEpoch for epoch in svConf.LR_EPOCH]
    lr_rate=[svConf.LR*(svConf.DECAY_FACTOR**x) for x in range(len(boundary)+1)]
    lr=tf.train.piecewise_constant(global_step,boundary,lr_rate)
    tf.summary.scalar('learn rate',lr)
    optimizer=tf.train.MomentumOptimizer(lr,0.9).minimize(loss,global_step)
    return optimizer

def start_train():
    prepare()
    # 第一步,获取输入
    image_batch, label_batch, roi_batch,landmark_batch = getInput()
    image_batch=image_color_distort(image_batch)

    # 第二部,搭建网络
    p_prob, p_regbox,p_landmark = buildModel(image_batch)

    if validate:
        image_batch_valid,label_batch_valid=validateInput()
        v_prob=validModel(image_batch_valid)
        valid_acc=validateAccuracy(v_prob,label_batch_valid)

    # 第三部,获得loss, class_loss,reg_loss,l2_loss,以及accuracy

    dis_total_loss, dis_acc = buildLoss(p_prob, p_regbox,p_landmark, label_batch, roi_batch,landmark_batch)
    # 第四步,定义训练算法,每次训练
    op_optimizer = buildTarget(dis_total_loss)
    op_summary = tf.summary.merge_all()

    # 开始训练

    MAX_STEPS = svConf.EPOCH *  svConf.LoopPerEpoch

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
                        writer.add_summary(_summary,global_step=i)
                        print('Total Loss is %.3f,Accuracy is %.3f' % (_loss, _acc))
                    if i % svConf.LoopPerEpoch == 0:
                        saver.save(sess, os.path.join(svConf.MODEL_CHECKPOINT_DIR,svConf.model_name),i // svConf.LoopPerEpoch)

                        #这里要交叉验证
                        if validate:
                            arr=[]
                            for kk in range(svConf.LoopsForValid):
                                _valid_acc=sess.run(valid_acc)
                                arr.append(_valid_acc)
                            _valid_acc=np.mean(arr)
                            print('Epoch %d,Valid acc is %.2f'%(i//svConf.LoopPerEpoch,_valid_acc))
            finally:
                coord.request_stop()

            coord.join(threads)
