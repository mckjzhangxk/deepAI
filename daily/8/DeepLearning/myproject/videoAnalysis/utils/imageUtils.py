import numpy as np
import torch
from torchvision import transforms as trans

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return torch.Tensor(y)

_standNormal = trans.Compose([
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
def uniformNormal(img,flip=False):
    if flip:
        img=trans.functional.hflip(img)
    return _standNormal(img)



# trans.functional.hflip
from PIL import Image
if __name__ == '__main__':
    I=Image.open('/home/zhangxk/projects/deepAI/daily/8/DeepLearning/myproject/videoAnalysis/data/multiface.jpg')
    print(uniformNormal(I).shape)
    print(uniformNormal(I,True).shape)