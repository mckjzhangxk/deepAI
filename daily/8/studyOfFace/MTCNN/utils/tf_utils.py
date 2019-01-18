import tensorflow as tf

'''
把一个tfrecord_file,转化成tensorflow的一个输入数据源,batch_size个.生成图片不同尺寸,归一化处理[-1,1]
返回:
    images:[batch_size,12,12,3]
    label:[batch]
    roi:[batch,4]         //相对于左上角,右下角修正
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
        'image/roi':tf.FixedLenFeature([4],tf.float32)
    })

    image=tf.decode_raw(image_example['image/encoded'],tf.uint8)
    image=(tf.cast(image,tf.float32)-127.5)/128.0
    image=tf.reshape(image,[imgsize,imgsize,3])
    label=tf.cast(image_example['image/label'],tf.float32)
    roi=tf.cast(image_example['image/roi'],tf.float32)


    image_batch,label_batch,roi_batch=tf.train.batch([image,label,roi],
                   batch_size=batchSize,
                   num_threads=1,
                   capacity=batchSize)
    return image_batch,label_batch,roi_batch


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