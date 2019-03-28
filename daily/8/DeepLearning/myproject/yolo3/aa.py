import tensorflow as tf
import numpy as np



def get_label_boxes(y):
    gh, gw = y.shape[0:2]
    logit = y[:, :, :, 4]  # (gh,gw,a)
    mask = logit > 0
    indices = np.where(mask)
    indices = np.array(indices)
    indices = indices.transpose()  # (#gt,3) 3=(i,j,a)

    # (#gt,5+C)
    labelobj = y[mask]

    y = gh * indices[:, 0]
    x = gw * indices[:, 1]

    labelobj_anchor = indices[:, 2]

    labelobj_cx = labelobj[:, 0] * gw + x
    labelobj_cy = labelobj[:, 1] * gh + y
    labelobj_w = labelobj[:, 2]
    labelobj_h = labelobj[:, 3]
    labelobj_classes = labelobj[:, 5:]

    labelobj_x1 = labelobj_cx - 0.5 * labelobj_w
    labelobj_y1 = labelobj_cy - 0.5 * labelobj_h
    labelobj_x2 = labelobj_cx + 0.5 * labelobj_w
    labelobj_y2 = labelobj_cy + 0.5 * labelobj_h

    return labelobj_x1,labelobj_y1,labelobj_x2,labelobj_y2,labelobj_anchor,labelobj_classes
y=np.random.rand(13,13,3,8)
y[...,4]=y[...,4]>0.5
labelobj_x1,labelobj_y1,labelobj_x2,labelobj_y2,labelobj_anchor,labelobj_classes=get_label_boxes(y)

print(labelobj_x1.shape)
print(labelobj_y1.shape)
print(labelobj_x2.shape)
print(labelobj_y2.shape)
print(labelobj_anchor.shape)
print(labelobj_classes.shape)
# print(np.sum(y[...,4]))
# print(np.prod(y.shape[:-1]))
# tf.enable_eager_execution()
# tf.executing_eagerly()
# line=tf.constant('/home/zhangxk/projects/deepAI/daily/8/DeepLearning/myproject/yolo3/data/raccoon_dataset/images/raccoon-163.jpg 6 7 240 157 0')
# print(line)
# sps = tf.string_split([line]).values
# path = sps[0]
# contents=tf.read_file(path)
# image = tf.image.decode_jpeg(contents, 3)
# image = tf.to_float(image)
# boxes = tf.string_to_number(sps[1:])
# boxes = tf.reshape(boxes, (-1, 5))
# print(contents)