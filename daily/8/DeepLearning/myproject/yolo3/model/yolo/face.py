from align.detect_face import detect_face,create_mtcnn
import tensorflow as tf
import cv2

sess=tf.Session()
pnet,rnet,onet=create_mtcnn(sess,None)

I=cv2.imread('')
sess=tf.Session()
pnet,rnet,onet=create_mtcnn(sess,None)
detect_face(I,30,pnet,rnet,onet,[0,6,0.6,0.7],0.7)