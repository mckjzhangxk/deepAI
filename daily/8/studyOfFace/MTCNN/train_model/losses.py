import tensorflow as tf

'''
输入的
prob可能是
    [None,1,1,2] //来自PNET
    [None,2]    //来自RNet,ONet
label:
    [None]      //标注,取值(-1,0,1)
label==0,1的参与计算loss

return:classification loss
'''
def classLosses(prob,label,eps=1e-10):
    '''
    为了得到稳定的计算结果,计算ls不采用直接公式:
        loss=mean(ylog(p)+(1-y)log(1-p))
    而是采用根据label的只是,选取参与计算loss
    的p
    '''
    prob=tf.squeeze(prob,[1,2])
    N=tf.shape(prob)[0]                 #获得样本个数
    ones = tf.ones(shape=[N,],dtype=tf.int32)
    zeros=tf.zeros(shape=[N,],dtype=tf.int32)

    prob=tf.reshape(prob,[2*N])   #拉直概率数组,odd表示对样本判断为-的概率,even表示判断为+
    idx=tf.range(tf.to_int32(N))*2 #根据上述建立索引

    #正样本是1,负,part是0
    label_filter=tf.where(tf.greater(label,0),ones,zeros)
    idx=idx+label_filter #默认选择的是[0,2,4,6,8]...也就是都是-的概率,如果label[i]是+,在idx[i]=idx[i]+1
    #用于计算loss的概率,PNET判定为label的概率!!!
    prob=tf.gather(prob,idx)

    pse_loss=-tf.log(prob+eps) #计算loss,其中label=-1的也参与了计算

    #有效的计算,只有0,1是有效的
    valid_label=tf.cast(tf.where(tf.greater(label,-1),ones,zeros),tf.float32)
    valid_num=tf.to_int32(tf.reduce_sum(valid_label))

    vaild_loss=valid_label*pse_loss

    select_loss,_=tf.nn.top_k(vaild_loss,valid_num)

    return tf.reduce_mean(select_loss)
'''
regbox:
    [None,1,1,4]//来自PNET
    [None,4] //来自RNet,ONet
roi:
    [None,4]  //标注
label=1,-1 的参与计算
return:
    regress loss,mse
'''
def boxesLoss(regbox,roi,label):
    regbox=tf.squeeze(regbox,[1,2])
    pse_loss=tf.reduce_sum((regbox-roi)**2,axis=1)


    N=tf.shape(roi)[0]
    ones= tf.ones(shape=[N])
    zeros=tf.zeros(shape=[N])
    #label=1,-1的是有效label
    valid_label=tf.where(tf.equal(label,0),zeros,ones)
    #有效数量
    valid_num=tf.to_int32(tf.reduce_sum(valid_label))

    valid_loss=pse_loss*valid_label
    valid_loss,_=tf.nn.top_k(valid_loss,k=valid_num)

    return tf.reduce_mean(valid_loss)


'''
统计 分类的正确性,注意,只统计label=1,0的
'''
def calAccuracy(prob,label):

    prob=tf.squeeze(prob,[1,2])
    logit=tf.cast(tf.arg_max(prob,1),tf.float32)

    #计算判断正确的总数,label=-1的一定对于0
    right_predict=tf.cast(tf.equal(label,logit),tf.float32)
    right_predict=tf.reduce_sum(right_predict)

    N=tf.shape(label)[0]
    ones=tf.ones(shape=[N])
    zeros=tf.zeros(shape=[N])

    #label=part(-1)的不是有效的
    valid_idx=tf.where(tf.equal(label,-1),zeros,ones)
    valid_num=tf.reduce_sum(valid_idx)

    acc=right_predict/valid_num
    return acc
