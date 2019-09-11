import torch
from yolov3.models1 import Upsample,Darknet


if __name__ == '__main__':

    x=torch.rand(1,3,416,416)
    cfg='cfg/yolov-obj.cfg'
    s=Darknet(cfg)

    s(x)
    # torch.jit.script(s)
    # print(s(x).shape)
    # script=torch.jit.trace(s,(x))
    # script.save('torchStript/xx.pt')