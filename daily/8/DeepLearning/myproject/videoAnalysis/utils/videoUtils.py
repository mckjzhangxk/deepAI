import cv2
import numpy as np

def readVideo(filename):
    cap=cv2.VideoCapture(filename)

    videoinfo={
        'frames':int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'codec':hex(int(cap.get(cv2.CAP_PROP_FOURCC))),
        'rate':cap.get(cv2.CAP_PROP_FPS),
        'width':int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height':int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'format':cap.get(cv2.CAP_PROP_FORMAT)
    }
    return cap,videoinfo

def videoWriter(output,videoformat='XVID',scale=(480,640),fps=30):
    codec=cv2.VideoWriter_fourcc(*videoformat)
    vid_writer = cv2.VideoWriter(output,codec,fps,scale)
    return vid_writer