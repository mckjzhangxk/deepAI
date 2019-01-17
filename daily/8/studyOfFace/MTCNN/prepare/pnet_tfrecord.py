from Configure import PNET_DATASET_PATH
from utils.tf_utils import _bytes_feature,_int64_feature,_float_feature

import os
import tensorflow as tf
import random
import cv2

'''
把一图片转化称为了bytes数组
'''
def _imagecode(imagepath):
    I=cv2.imread(imagepath)
    assert len(I.shape)==3 ,'invalid image!'
    assert I.shape[2]==3,' image must have 3 channels!'
    I=I[:,:,::-1]
    return I.tostring()
'''
record:是一行记录,空格分开
    imagepath label rx1 ry1 yx2 ry2

转化成了TF记录,写入文件!
'''
def _add_to_TFRecord(record,writer):
    sps=record.split(' ')
    imagepath=sps[0]

    imagedata=_imagecode(imagepath)
    label=int(sps[1])
    roi=list(map(float,sps[2:]))

    example=tf.train.Example(features=tf.train.Features(feature={
        'image/encoded':_bytes_feature(imagedata),
        'image/label':_int64_feature(label),
        'image/roi':_float_feature(roi)
    }))
    writer.write(example.SerializeToString())

def cvtTxt2TF(shuffle=True):
    with open(os.path.join(PNET_DATASET_PATH,'PNet.txt')) as fs:
        records = fs.readlines()
        if shuffle:
            random.shuffle(records)


    #输出TF文件名称
    tf_filename=os.path.join(PNET_DATASET_PATH,'PNet_shuffle')
    if tf.gfile.Exists(tf_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return

    with tf.io.TFRecordWriter(tf_filename) as writer:
        for idx,x in enumerate(records):
            _add_to_TFRecord(x,writer)
            if (idx+1)%100==0:
                print('>>%d/%d images has been converted'%(idx + 1,len(records)))

if __name__ == '__main__':
    cvtTxt2TF(True)
