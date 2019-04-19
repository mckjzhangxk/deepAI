import tensorflow as tf

from model.yolo.DarkNet import darknet53


class ImageDataset():
    def __init__(self,imagesize):
        self.imagesize=imagesize
    def build_example(self, filepath, batch_size=32, parallels=1, eager=False):
        '''

        :param filepath: 
        :param batch_size: 
        :param epoch: 
        :return: 
        '''
        dataset = tf.data.TextLineDataset(filepath)

        dataset = dataset.map(self._retrive, parallels)
        dataset = dataset.map(self._resizeImage, parallels)


        dataset = dataset.repeat(1)
        dataset = dataset.batch(batch_size)

        iterator = dataset.make_one_shot_iterator()
        if eager: return iterator

        image= iterator.get_next()
        return image

    def _retrive(self, line):
        sps = tf.string_split([line]).values
        path = sps[0]
        content = tf.read_file(path)
        image = tf.image.decode_jpeg(content, 3)
        image = tf.to_float(image) / 255.0
        return image

    def _resizeImage(self, image):
        image = tf.image.resize_images(image, (self.imagesize, self.imagesize))
        return image

db=ImageDataset(416)
X=db.build_example('/home/zxk/AI/coco/annotations/train.txt',batch_size=48,eager=False)
model=darknet53(X)
model.extractFeature('/home/zxk/AI/tensorflow-yolov3/checkpoint/yolov3.ckpt',
                     '/home/zxk/AI/coco/annotations/train_darknet53.rfrecords')
