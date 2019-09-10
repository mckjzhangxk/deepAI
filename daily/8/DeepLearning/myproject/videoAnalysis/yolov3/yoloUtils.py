import torch
import os
from  yolov3.models import Darknet,load_darknet_weights
from yolov3.utils.utils import non_max_suppression
import cv2


class CCPD_YOLO_Detector():
    def __init__(self,cfg=None,weight=None,img_size=(416,416),device=None,half=False):
        if cfg==None:
            cfg=os.path.dirname(os.path.abspath(__file__))
            cfg=os.path.join(cfg,'cfg/yolov-obj.cfg')
        if weight==None:
            weight = os.path.dirname(os.path.abspath(__file__))
            weight = os.path.join(weight, 'cfg/yolov-obj_final.weights')

        assert os.path.exists(cfg),'yolo.configure file must exist'
        assert os.path.exists(weight),'yolo.weight file must exist'

        self.img_size=img_size

        model=Darknet(cfg,img_size)
        load_darknet_weights(model,weight)
        model.fuse()

        self.dtype='torch.FloatTensor'
        if half:
            model=model.half()
            self.dtype='torch.HalfTensor'
        self.model=model.to(device)
        self.model.eval()
        print('load detector weight of %s'%weight)

        self.device=device
    def _image2Tensor_(self,Is):
        '''
        
        :param Is:list of numpy image,BGR
        :return: tensor(batch,3,self.img_size,self.img_size)
        '''
        ret=[]
        for I in Is:
            I = cv2.resize(I, self.img_size)
            I=cv2.cvtColor(I,cv2.COLOR_BGR2RGB).transpose(2,0,1)/255.0
            x = torch.from_numpy(I)
            ret.append(x)
        ret=torch.stack(ret,0).type(self.dtype).to(self.device)


        return ret

    def predict(self,I,conf_thres=.5,nms_thres=.5):
        '''

        :param I:list of cv2.imread,BGR format
        :param conf_thres:
        :param nms_thres:
        :return:list(list(tuple6)),和I有一样的长度，out[i]对应第i个图片的结果 
        结果=list of (x1,y1,x2,y2,confident,label)
        没有结果，这个list 就是[]
        '''
        if not isinstance(I,list):
            I=[I]

        oldshapes=[x.shape for x in I]

        x=self._image2Tensor_(I)
        y,_=self.model(x)
        dets = non_max_suppression(y.float(),conf_thres,nms_thres)

        ret=[]
        for det,oldshape in zip(dets,oldshapes):
            current_result=[]
            if det is not None and len(det)>0:
                H,W,_=oldshape
                for (x1,y1,x2,y2,score,class_conf,cls)in det:
                    x1=max(min((x1/self.img_size[1])*W,W),0)
                    x2=max(min((x2/self.img_size[1])*W,W),0)

                    y1=max(min((y1/self.img_size[0]*H),H),0)
                    y2=max(min((y2/self.img_size[0]*H),H),0)
                    current_result.append((int(x1),int(y1),int(x2),int(y2),float(score*class_conf),int(cls)))
            ret.append(current_result)
        return ret


if __name__ == '__main__':
    import cv2

    det=CCPD_YOLO_Detector()

    I=cv2.imread('/home/zhangxk/projects/deepAI/daily/8/DeepLearning/myproject/videoAnalysis/RetinaFace/2.jpg')
    det.predict()