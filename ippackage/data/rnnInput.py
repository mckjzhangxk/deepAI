import numpy as np
import tensorflow as tf
import collections

class TrainInput(collections.namedtuple('TrainInput',['X','Y','X_len','Cursor','Iterator','Update_Source'])):pass

def _dataSource(datafile, BATCH_SIZE, D):
    '''
        datafile:要训练的数据集,每一行是一个EXAMPLE,第一个字段是label,第二个字段是times,
        其他的是T个序列的特征,换句话说,后面的字段长度是T*D
        
        BATCH_SIZE,D:batch,特征数量
        
        本方法把数据集进行了
            1.分割
            2.标签,时序长度,特征分割
            3.类型转化
            5.文件shape转化
        返回(source,label,source_length,iterator)
        
        source:(N,T,D,float32),N=BATCH_SIZE,在文件结尾的位置<BATCH_SIZE
            T:examples的最大长度
        label:(N,int32):样本的标签1-FREEGATE,2-VPN,0-default
        source_length:(N,int32)每一个样本的实际长度
        iterator:初始化数据集用的sess.run(iterator.initializer)
    '''


    dataset=tf.data.TextLineDataset(datafile)

    #分割字符串
    dataset=dataset.map(lambda x:tf.string_split([x],',').values)
    #第一个字段是label,第二个是seq_length,其他是时序的特征
    dataset=dataset.map(lambda x:(x[0],x[1],x[2:]))
    #label->int,features->float32
    dataset=dataset.map(lambda label,times,features:(
                tf.string_to_number(label,tf.int32),
                tf.string_to_number(times, tf.int32),
                tf.string_to_number(features,tf.float32))
                   )
    #label,features->label,features,label,size(features)

    dataset=dataset.batch(BATCH_SIZE)

    iterator=dataset.make_initializable_iterator()
    label,source_length,source=iterator.get_next()
    batch_size=tf.shape(source_length)[0]
    source=tf.reshape(source,[batch_size,-1,D])


    return (source,label,source_length,iterator)

def _assign(source,label,sourceLen,BATCH,Tmax,D):
    '''
    source(batch,Tmax,D,float32):是从源文件取出批量时序训练数据,batch在文件结尾的时候<BATCH
    label(batch,int32):训练数据的标注
    sourceLen(batch,int32):一条训练数据的实际长度,一般<=Tmax
    
    
    创建4个变量
    batch_size:(int32)
    X(BATCH,Tmax,float32):
    Y:(BATCH,int32)
    X_len:(BATCH,int32)
    
    然后定义一批复制操作
        shape(source)---------->batch_size
        source[:batch_size]---->X
        label[:batch]---------->Y
        sourceLen[:batch]------>X_len
        合并上面的复制操作为assign_op
        
        执行assign_op意味着更新数据源!
    返回    
        assign_op,X,Y,X_lenbatch_size
    :param source: 
    :param label: 
    :param sourceLen: 
    :param BATCH: 
    :param Tmax: 
    :param D: 
    :return: 
    '''
    with tf.variable_scope('INPUT',initializer=tf.zeros_initializer):
        Xsource=tf.get_variable('Xsource',shape=[BATCH,Tmax,D],dtype=tf.float32,trainable=False)
        Ysource=tf.get_variable('Ysource',shape=[BATCH],dtype=tf.int32,trainable=False)
        XsourceLen=tf.get_variable('XsourceLen',shape=[BATCH],dtype=tf.int32,trainable=False)
        #保存实际的批量
        batch_size=tf.get_variable('batch_size',shape=(),dtype=tf.int32,trainable=False)


        shape=tf.shape(source)
        N=shape[0]
        assign_op1=tf.assign(Xsource[:N],source)
        assign_op2=tf.assign(Ysource[:N], label)
        assign_op3=tf.assign(XsourceLen[:N],sourceLen)
        assign_op4 = tf.assign(batch_size, N)

        assign_op=tf.group([assign_op1,assign_op2,assign_op3,assign_op4])

    return (assign_op,Xsource,Ysource,XsourceLen,batch_size)

def _get_cursor(p,Tmax):
    '''
    P:int,cursor每次增长量,增到Tmax后会回到0
    创建变量cursor保存当前位置,和更新操作pointer
    每次调用pointer,cursor都会+p,超过Tmax会归入0,表示你遍历完成一组训练集
    返回(pointor,cursor)
    :param p: 
    :return: 
    '''
    with tf.variable_scope('Input'):
        cursor = tf.get_variable('cursor', shape=(), dtype=tf.int32,
                                 trainable=False,
                                 initializer=tf.constant_initializer(-p))
        pointor = tf.assign(cursor,(cursor+p)%(Tmax), name='pointor')

        return pointor, cursor


def get_input(datafile,BATCH_SIZE,Tmax,D,perodic):
    '''
    datafile:训练的数据来源
    BATCH_SIZE:输入数据的批量
    Tmax:时序的长度
    D:特征维度
    perodic:输入到RNN的序列 时序维度,注意 Tmax//perodic
    
    把datafile转成source(batchsize,Tmax,D),label(batchsize),source_len(batchsize)
    
    然后定义一个更新数据源的操作update_source,每次执行update_source,就会把source,labelsource_len,
    的新值到存储到寄存器:
        Xsource(BATCH_SIZE,Tmax,D),Ysource(BATCH_SIZE),Xsource_len(Batch),batchsize中,
    你可以对这些取出数据任意操作,数据源数据不会变化,除非执行update_source
    
    定义pointor,curosr,pointer是取以一个perodic个时序的错在,cursor表示当前时序在Xsource的位置,
    和Xsource_len联合使用 可以指定是否是有效时序
    
    返回RNN的输入
        X(batch,perodic,D,float32):=Xsource[:batch,pointor*Perodic:pointor+1*Perodic]
        Y(batch,int32):=Ysource[:batch]
        X_len(batch,int32):=Xsource_len[:batch]
        cursor:tensor,int32,当前指针的位置
        update_source_op:刷新数据源的操作
        iterator:重置数据源的操作
    :param datafile: 
    :param BATCH_SIZE: 
    :param Tmax: 
    :param D: 
    :param perodic: 
    :return: 
    '''

    #获得数据源
    source,tgt,source_len,iterator=_dataSource(datafile,BATCH_SIZE,D)
    #寄存数据源
    update_source_op,R_source,R_tgt,R_source_len,R_batchsize=\
        _assign(source,tgt,source_len,BATCH_SIZE,Tmax,D)
    #获得指针
    pointer,cursor=_get_cursor(perodic,Tmax)

    #
    X=R_source[:R_batchsize,pointer:pointer+perodic]
    Y=R_tgt[:R_batchsize]
    X_len=R_source_len[:R_batchsize]

    return TrainInput(X,Y,X_len,cursor,iterator,update_source_op)
filename='/home/zhangxk/projects/deepAI/ippackage/data/data'
BATCH,T,D=3,6,2
perodic=3


trainInput=get_input(filename,BATCH,T,D,perodic)

sess=tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(trainInput.Iterator.initializer)

steps=T//perodic
for i in range(2):
    sess.run(trainInput.Update_Source)

    for s in range(steps):
        _x,_y,_xl,_cursor=sess.run([trainInput.X,trainInput.Y,trainInput.X_len,trainInput.Cursor])
        print('x:',_x)
        print('y:', _y)
        print('len:', _xl)
        print('_cursor:', _cursor)
        print('-------------------')
    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

sess.close()
