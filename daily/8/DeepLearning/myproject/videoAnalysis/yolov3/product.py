import torch
import torch.nn as nn
from yolov3.models1 import Upsample,Darknet,create_modules,EmptyLayer,YOLOLayer,load_darknet_weights
from yolov3.utils.parse_config import parse_model_cfg


def testSave(filename,model,inputx):
    script=torch.jit.trace(model,inputx)
    script.save(filename)
class MyDarkNet(nn.Module):
    def __init__(self):
        super(MyDarkNet,self).__init__()
        cfg = 'cfg/yolov-obj.cfg'

        self.model_defs = parse_model_cfg(cfg)
        self.model_defs[0]['cfg'] = cfg

        self.hparam, self.model_list = create_modules(self.model_defs)

    def forward(self,x):
        layoutput = []
        output=[]
        for i, (model_def, model) in enumerate(zip(self.model_defs, self.model_list)):
            mtype = model_def['type']
            if mtype in ['convolutional', 'upsample', 'maxpool']:
                x = model(x)
            elif mtype == 'route':
                layers = [int(x) for x in model_def['layers'].split(',')]
                if len(layers) == 1:
                    x = layoutput[layers[0]]
                else:
                    x = torch.cat([layoutput[ll] for ll in layers], dim=1)
            elif mtype == 'shortcut':
                layer_i = int(model_def['from'])
                x = layoutput[-1] + layoutput[layer_i]
            elif mtype == 'yolo':
                x = model(x)
                output.append(x)
            layoutput.append(x)

        xx=list(zip(*output))
        io=torch.cat(xx[0],1)
        return io

if __name__ == '__main__':

    cfg='cfg/yolov-obj.cfg'
    weight='cfg/yolov-obj_final.weights'


    x = torch.randn(1, 3, 416,416)
    model=Darknet(cfg='cfg/yolov-obj.cfg')
    load_darknet_weights(model, weight)

    model.fuse()
    model.eval()


    script=torch.jit.trace(model,x)
    script.save('torchStript/yolonet.pt')

