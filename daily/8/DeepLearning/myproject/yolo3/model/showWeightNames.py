import tensorflow as tf
import tensorflow.contrib as tfcontrib


def showCkpt(path):
    reader=tfcontrib.framework.load_checkpoint(path)
    varname_shape=reader.get_variable_to_shape_map()
    ll=sorted(varname_shape.items(),key=lambda x:x[0])

    for k,v in ll:
        print(k,v)
showCkpt('/home/zhangxk/projects/deepAI/daily/8/DeepLearning/myproject/yolo3/modelzoo/weights/yolov3.ckpt')