from recognize.model import CNet
import torch
import os
import cv2


provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']
def decode(Y):
    P=provinces[Y[0]]
    S=alphabets[Y[1]]
    C=[ads[y] for y in Y[2:]]
    return P+S+''.join(C)

class CCPD_Recognizer():

    def __init__(self,modelpath=None,device=None,*agrs,**kwargs):
        '''

        :param agrs:
        :param kwargs:
        '''

        if modelpath==None:
            basedir=os.path.dirname(os.path.abspath(__file__))
            modelpath=os.path.join(basedir,'models/E14.pt')

        self.device=device
        self.reg_model=CNet().to(device)
        self._initial_model_(modelpath)
        self.reg_model.eval()
    def _initial_model_(self,path):
        assert  os.path.exists(path),"model path %s do not exist"%path

        checkpoint=torch.load(path)
        self.reg_model.load_state_dict(checkpoint['model'])

        print('load model from path:{}'.format(path))

    def predict(self,I):
        shape=I.shape
        if shape[-1]==3:
            I=cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
        I=cv2.resize(I,(64,32))/255.0
        I=I[None]

        x=torch.from_numpy(I)[None].to(self.device).float()
        yhat=self.reg_model(x)[0]
        yhat=torch.split(yhat,[34,25,35,35,35,35,35])


        result=[]
        for c in yhat:
            code=torch.argmax(c).data.cpu().item()
            result.append(code)

        resultStr=decode(result)

        return resultStr