import tensorflow as tf
import tensorflow.contrib as tfcontrib
import collections
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split




class BatchInput(collections.namedtuple('BatchInput',['X','Y','initializer'])):pass


def imageshow(X,Y):
    plt.figure()
    N=len(X)
    for n in range(N):
        plt.subplot(N,2,2*n+1)
        plt.imshow(X[n])
        plt.subplot(N,2,2*n+2)
        plt.imshow(Y[n])
    plt.show()
def prepare_dataset(hparam):

    with tf.gfile.GFile(hparam.train_mask_file) as fs:
        lines=fs.readlines()[1:]
        filenames=list(map(lambda line:line.split(',')[0].split('.')[0],lines))

    src_files=list(map(lambda line:os.path.join(hparam.src_prefix,line+'.jpg'),filenames))
    tgt_files=list(map(lambda line:os.path.join(hparam.tgt_prefix,line+'_mask.gif'),filenames))

    train_src_files,test_src_files,train_tgt_file,test_tgt_file=\
        train_test_split(src_files,tgt_files,train_size=hparam.train_size)

    train_batch=get_iterator(hparam,train_src_files,train_tgt_file,'train')
    test_batch=get_iterator(hparam,test_src_files,test_tgt_file,'eval')

    return train_batch,test_batch

def get_iterator(hparam,src_files,tgt_files,mode='train'):
    '''
    对src_files，tgt_files的数据进行相同的 反转，平移，hue调节操作。
    src_files:list of input file_paths
    tgt_files:list of output_file_paths
    
    返回:BatchIuput:
        input:(batch,H,W,3)
        label:(batch,H,W,1)
    :param hparam: 
    :param src_files: 
    :param tgt_files: 
    :return: 
    '''
    def _decode_image(src,tgt):
        src=tf.read_file(src)
        tgt=tf.read_file(tgt)

        src=tf.image.decode_jpeg(src,3)
        tgt=tf.image.decode_gif(tgt)[0][:,:,0:1]
        return src,tgt
    def _resize_image(src,tgt):
        src=tf.image.resize_images(src,hparam.image_size)
        tgt=tf.image.resize_images(tgt,hparam.image_size)
        return src,tgt

    def _flip_image(src,tgt):
        return tf.cond(tf.random_uniform(())<0.5,
                lambda :(src,tgt),
                lambda :(tf.image.flip_left_right(src),tf.image.flip_left_right(tgt)))
    def _shift_image(src,tgt):
        shape=tf.shape(src)
        H,W=tf.to_float(shape[0]),tf.to_float(shape[1])
        r=tf.random_uniform((),-hparam.shift_range,hparam.shift_range)
        dH,dW=r*H,r*W

        src=tfcontrib.image.translate(src,[dW,dH])
        tgt=tfcontrib.image.translate(tgt, [dW, dH])
        return src,tgt
    def _hue_image(src,tgt):
        src=tf.image.random_hue(src,hparam.hue_delta)
        return src, tgt



    dataset=tf.data.Dataset.from_tensor_slices((src_files,tgt_files))
    dataset=dataset.map(_decode_image,num_parallel_calls=hparam.num_parallel_calls)

    #图片尺寸变化
    dataset=dataset.map(_resize_image,num_parallel_calls=hparam.num_parallel_calls)
    #数据加强，反转
    dataset=dataset.map(_flip_image,num_parallel_calls=hparam.num_parallel_calls)

    #平移
    dataset = dataset.map(_shift_image, num_parallel_calls=hparam.num_parallel_calls)

    #hue
    dataset=dataset.map(_hue_image,num_parallel_calls=hparam.num_parallel_calls)

    dataset=dataset.batch(hparam.batch_size)
    if mode=='train':
        dataset=dataset.repeat(hparam.epoch)
    iter=dataset.make_initializable_iterator()
    #shape (batch,H,W,3)
    imgs,labels=iter.get_next()
    imgs=tf.to_float(imgs)*hparam.scale
    labels=tf.to_float(labels)*hparam.scale

    return BatchInput(imgs,labels,iter.initializer)
# import numpy  as np
# train_batch,test_batch=prepare_dataset(hparam)
# print(train_batch)
# with tf.Session() as sess:
#     sess.run(train_batch.initializer)
#     for i in range(3):
#         X,Y=sess.run([train_batch.X,train_batch.Y])
#         X=X.astype('uint8')
#         Y = Y.astype('uint8')
#         imageshow(X,Y[:,:,:,0])