import tensorflow as tf

class FeatureWriter(object):
  """Writes InputFeature to TF example file."""

    def __init__(self, filename, is_training):
        self.filename = filename
        self.is_training = is_training
        self.num_features = 0
        self._writer = tf.io.TFRecordWriter(filename)
    def process_feature(self, feature):
        '''
        
        :param feature:字典,key是特征名,value是特征值
        :return: 
        '''
        pass
    def close(self):
        self._writer.close()