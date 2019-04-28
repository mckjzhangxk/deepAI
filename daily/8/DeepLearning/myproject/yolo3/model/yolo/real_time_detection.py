# coding=utf-8
"""Performs face detection in realtime.

Based on code from https://github.com/shanren7/real_time_face_recognition
"""
# MIT License
#
# Copyright (c) 2017 François Gervais
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import argparse
import sys
import time

import cv2
import tensorflow as tf
from align.detect_face import detect_face,create_mtcnn
from model.yolo.YoloNet import YoLoService
import numpy as np


def add_overlays(frame, faces, frame_rate):
    for box,score,label in zip(faces['boxes'],faces['scores'],faces['labels']):
        box=box.astype(int)
        cv2.rectangle(frame,
                      (box[0], box[1]), (box[2], box[3]),
                      (0, 255, 0), 2)
        cv2.putText(frame,str(score), (box[0], box[3]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                    thickness=2, lineType=2)

    cv2.putText(frame, str(frame_rate) + " fps", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                thickness=2, lineType=2)

def cropPeople(frame,objs,frameIdx,output_dir):
    def outoutprefixname():
        a=frameIdx//25
        b=frameIdx%25
        return '%d-%d'%(a,b)

    boxes,labels,scores=objs['boxes'].astype(np.int),objs['labels'],objs['scores']
    prefix=outoutprefixname()
    for i,label in enumerate(labels):
        if label==1:
            box=boxes[i]
            Icrop=frame[box[1]:box[3],box[0]:box[2]]

            cv2.imwrite(output_dir+'/%s-%d.jpg'%(prefix,i),Icrop)
def main():
    frame_interval = 3  # Number of frames after which to run face detection
    model_path = '/home/zxk/AI/tensorflow-yolov3/checkpoint/yolov3.ckpt'

    # detector=YoLoService(model_path)
    sess = tf.Session()
    pnet, rnet, onet = create_mtcnn(sess, None)

    frame_count = 0

    video_capture = cv2.VideoCapture('/home/zxk/AI/y2mate.com - _c0raDbZpV9s_360p.mp4')
    # face_recognition = face.Recognition()


    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if not ret:break

        frame_count+=1
        if frame_count%1==0:
            # frame=cv2.rotate(frame,cv2.ROTATE_90_CLOCKWISE)
            frameorgin=frame
            H,W,_=frameorgin.shape


            frame=cv2.resize(frame,(640,480),interpolation=cv2.INTER_LINEAR)
            # objects=detector.predict(frame)
            total_boxes,_=detect_face(frame[:,:,::-1], 30, pnet, rnet, onet, [0.6, 0.6, 0.7], 0.7)
            objects={'boxes':total_boxes,'labels':[1]*len(total_boxes),'scores':[1]*len(total_boxes)}
            # cropPeople(frameorgin,objects,frame_count,'out')

            add_overlays(frame,objects,1)
            cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        print(frame_count,ret)
    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()