import tensorflow as tf
import cv2
import random
import os
from utils.common import progess_print
'''

从原文件到目标文件的输出
原文件:.txt文件
目标文件:TFRECORD 文件
'''
def cvtTxt2TF(BASE_DIR, srcname, dscname, shuffle=True,display=100):
    print('convert %s-------->%s'%(srcname,dscname))
    with open(os.path.join(BASE_DIR, srcname)) as fs:
        records = fs.readlines()
        if shuffle:
            random.shuffle(records)


    #输出TF文件名称
    tf_filename=os.path.join(BASE_DIR, dscname)
    if tf.gfile.Exists(tf_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return

    with tf.io.TFRecordWriter(tf_filename) as writer:
        for idx,x in enumerate(records):
            writeTFRecord(x, writer)
            if (idx+1)%display==0:
                progess_print('%d/%d record have been converted'%(idx+1,len(records)))

'''
把一个tfrecord_file,转化成tensorflow的一个输入数据源,batch_size个.生成图片不同尺寸,归一化处理[-1,1]
返回:
    images:[batch_size,12,12,3]
    label:[batch]
    roi:[batch,4]         //相对于左上角,右下角修正
    landmark:[batch,10]
    想象成image=tf.placeHolder([bs,12,12,3]).....
    例如,1 0.09 -0.19 0.02 -0.05:表示存在人脸,人脸在(0.09,-0.19),(1.02,0.95)的box(修正(0,0),(1,1))内 
'''
def readTFRecord(tf_file,batchSize,imgsize=12):
    queue=tf.train.string_input_producer([tf_file],shuffle=True)
    reader=tf.TFRecordReader()
    _,serialLine=reader.read(queue)

    image_example=tf.parse_single_example(
        serialLine,
        features={
        'image/encoded':tf.FixedLenFeature([],tf.string),
        'image/label':tf.FixedLenFeature([],tf.int64),
        'image/roi':tf.FixedLenFeature([4],tf.float32),
        'image/landmark': tf.FixedLenFeature([10], tf.float32)
    })

    image=tf.decode_raw(image_example['image/encoded'],tf.uint8)
    image=(tf.cast(image,tf.float32)-127.5)/128.0
    image=tf.reshape(image,[imgsize,imgsize,3])
    label=tf.cast(image_example['image/label'],tf.float32)
    roi=tf.cast(image_example['image/roi'],tf.float32)
    landmark=tf.cast(image_example['image/landmark'],tf.float32)

    image_batch,label_batch,roi_batch,landmark_batch=tf.train.batch([image,label,roi,landmark],
                   batch_size=batchSize,
                   num_threads=1,
                   capacity=batchSize)
    return image_batch,label_batch,roi_batch,landmark_batch


'''
record:是一行记录,空格分开
    imagepath label rx1 ry1 yx2 ry2

转化成了TF记录,写入文件!
'''

def writeTFRecord(record, writer):
    '''
    把一图片转化称为了bytes数组
    '''

    def _imagecode(imagepath):
        I = cv2.imread(imagepath)
        assert len(I.shape) == 3, 'invalid image!'
        assert I.shape[2] == 3, ' image must have 3 channels!'
        I = I[:, :, ::-1]
        return I.tostring()

    sps = record.split(' ')
    imagepath = sps[0]

    imagedata = _imagecode(imagepath)
    label = int(sps[1])
    roi = list(map(float, sps[2:6]))
    landmark=list(map(float, sps[6:]))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': _bytes_feature(imagedata),
        'image/label': _int64_feature(label),
        'image/roi': _float_feature(roi),
        'image/landmark':_float_feature(landmark)
    }))
    writer.write(example.SerializeToString())

def _int64_feature(value):
    """Wrapper for insert int64 feature into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for insert float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for insert bytes features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))