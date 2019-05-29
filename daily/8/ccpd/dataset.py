import glob
from torch.utils.data import DataLoader,Dataset
import PIL.Image as Image
import numpy as np
import torchvision.transforms as transforms
import os
import torch

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

class CCPD_Dataset(Dataset):
    
    def __init__(self,basedir,train,seed=0):
        super().__init__()   
        db=glob.glob(basedir)
        n=len(db)
        np.random.seed(seed)
        totalidx=set(np.arange(n))
        trainidx=set(np.random.choice(n,int(n*0.98),False))
        testidx=totalidx-trainidx
        
        if train:
            self._db=[db[idx] for idx in trainidx]
        else:
            self._db=[db[idx] for idx in testidx]
        self._init_processing_fn()
    def _init_processing_fn(self):

        t_affine=transforms.RandomAffine(5,(0.05,0.05),(0.9,1.1))
        t_color=transforms.ColorJitter(brightness=.3, contrast=.3)
        t_resize=transforms.Resize((32,64))
        t_tensor=transforms.ToTensor()
        
        t_norm=transforms.Lambda(lambda t:(t-0.5)/0.5)
        
        self.process=transforms.Compose([t_color,t_affine,t_resize,t_tensor])
        
        
#         self._db=['/home/zhangxk/AIProject/CCPD/sample/rotate/025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg']
    def __getitem__(self,k):
        imagename=self._db[k]  
        basename=os.path.basename(imagename)
        sps=basename.split('-')
        #label
        label=sps[4]
        labels=label.split('_')
        Y=np.array([int(k) for k in labels])
        #image
#         "025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg"
        bbox=sps[2]
        bbox=bbox.split('_')
        pp1=bbox[0].split('&')
        pp2=bbox[1].split('&')
        x1,y1,x2,y2=int(pp1[0]),int(pp1[1]),int(pp2[0]),int(pp2[1])

        I=Image.open(imagename,mode='r').convert("L")
        X=I.crop((x1,y1,x2,y2))
        return self.process(X),torch.tensor(Y)
    def __len__(self):
        return len(self._db)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dbpath='/home/zxk/AI/data/CCPD2019/ccpd_base/*.jpg'
    trainset=CCPD_Dataset(dbpath,True)
    testset=CCPD_Dataset(dbpath,False)
    print('trainset:',len(trainset))
    print('teset:',len(testset))
    batch_size=64
    trainloader=DataLoader(trainset,batch_size,shuffle=True,num_workers=4)
    testloader=DataLoader(trainset,batch_size,shuffle=False,num_workers=4)
    #########test code#############
    N=8765
    ds=trainset
    print(ds[N][0].squeeze(0).numpy().shape)
    plt.imshow(ds[N][0].squeeze(0).numpy(),cmap='gray')
    plt.show()

    print(decode(ds[N][1]))
    for k in trainloader:
        print(k[0].size(),k[1].size())
        break