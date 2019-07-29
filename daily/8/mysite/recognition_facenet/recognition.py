from scipy.misc import imread,imresize
from scipy import misc
from facenet.align.detect_face import detect_face,create_mtcnn

import numpy as np
import tensorflow as tf
import facenet.facenet as facenet
import matplotlib.pyplot as plt
from recognition_facenet.ImageProcess import modify_autoContract,equalization
import os


def face_encodings(filename,imageSize=160,pad=22,prefunc=None):
    I=_detectSingleFace(filename,image_size=imageSize,pad=pad,preprocsingFunc=prefunc)
    if I is None:return []
    
    if I.shape!=4:
        I.shape=(-1,imageSize,imageSize,3)
    code=model.predict(I)
    return [code[0]]

def _detectSingleFace(path, minsize=50, threshold = [0.6, 0.7, 0.7], factor=0.709,pad=5,image_size=160,preprocsingFunc=None):

    I=imread(path,mode='RGB')
    if preprocsingFunc:
        I=preprocsingFunc(I)
    box, point=detect_face(I,minsize,pnet_fun,rnet_fun,onet_fun,threshold,factor)

    if len(box)==0:
        return None

    h,w=I.shape[0:2]
    x1, y1, x2, y2, acc = box[0]
    x1, y1, x2, y2 = max(x1-pad, 0), max(y1-pad, 0), min(x2+pad, w), min(y2+pad, h)
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    cropped=I[y1:y2,x1:x2]
    aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
    prewhitened=facenet.prewhiten(aligned)

    return prewhitened

def _getModel():

    basepath=os.path.split(os.path.realpath(__file__))[0]
    model_path=os.path.join(basepath,'model')
    print('load model from %s'%model_path)
    with sess.as_default():
        facenet.load_model(model_path)

        # Get input and output tensors
        images_placeholder = sess.graph.get_tensor_by_name("input:0")
        embeddings = sess.graph.get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        class RetModel():
            def predict(self,image_input):
                '''
                
                :param image_input: should have shape nx160x160x3
                :return: nx512
                '''
                feed_dict={images_placeholder:image_input,phase_train_placeholder:False}
                emb=sess.run(embeddings,feed_dict=feed_dict)
                return emb

    return RetModel()
sess=tf.Session()
# #detection network
# pnet_fun, rnet_fun, onet_fun = create_mtcnn(sess,None)
# #recognition network

model=_getModel()
# writer=tf.summary.FileWriter('model',sess.graph)
# writer.flush()

import cv2
if __name__ == '__main__':
    def prewhiten(x):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1 / std_adj)
        return y
    I=cv2.imread('/home/zxk/PycharmProjects/deepAI1/daily/8/mysite/webapp/facedb1/4.png')
    I = I[:, :, ::-1]
    I=cv2.resize(I,(160,160))
    I = I[None]

    I = prewhiten(I)
    xx=model.predict(I)
    print(xx[0])
#     num=0
#     for v in tf.trainable_variables():
#         num+=np.product(v.get_shape())
#     print(num)
    # ipath='/home/zhangxk/projects/myfaceproject/facedb/'
    #
    # for f in os.listdir(ipath):
    #     I=_detectSingleFace(ipath+f)
    #     code=face_encodings(ipath+f)
    #     if len(code)==0:
    #         print('%s, can not find face'%f)
    #     else:
    #         print(code[0].shape)
    # print('xxxxxxxxxxxxxxx')