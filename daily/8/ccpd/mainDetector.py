from recognize.ccptUtils import CCPD_Recognizer
from yolov3.yoloUtils import CCPD_YOLO_Detector
import torch


class CCPD_Detector():


    def __init__(self,
                 detector_cfg=None,
                 detector_weight=None,
                 detector_image_size=(416,416),
                 recognizer_path=None):
        assert torch.cuda.is_available(), "this application must run on GPU"
        self.device=torch.device('cuda')

        self.detector=CCPD_YOLO_Detector(cfg=detector_cfg,
                                         weight=detector_weight,
                                         img_size=detector_image_size,
                                         device=self.device)
        self.recognizer=CCPD_Recognizer(modelpath=recognizer_path,
                                        device=self.device)

    def predict(self,I):
        '''

        :param I:cv2.imread,BGR format
        :return:
        '''

        det=self.detector.predict(I)
        ret=[]
        for (x1,y1,x2,y2,score,cls) in det:
            pstr=self.recognizer.predict(I[int(y1):int(y2),int(x1):int(x2)])
            ret.append((x1,y1,x2,y2,score,cls,pstr))
        return ret
