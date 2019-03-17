import tensorflow as tf
import numpy as np
with tf.gfile.FastGFile('../data/train.vi') as fs:
    lines=fs.readlines()

    arr=[]
    for l in lines:
        arr.append(len(l.split()))
        # print(l)
        # print(l.split())
    print(max(arr))
    print(np.mean(arr))
    print(np.median(arr))