import tensorflow as tf
import numpy as np
from dbutils import ImageBrower


tf.enable_eager_execution()
tf.executing_eagerly()

imgbrower=ImageBrower('data/sample.txt','data/raccoon_my_anchors.txt',C=1)
for i in range(2):
    image, image13, image26, image52=next(imgbrower)
    print(image.shape,image13.shape,image26.shape,image52.shape)