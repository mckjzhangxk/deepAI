# https://susanqq.github.io/UTKFace/
import os
import glob
from torch.utils.data import DataLoader,Dataset
import torch
from facenet_pytorch import InceptionResnetV1
from utils.imageUtils import prewhiten
import cv2
import numpy as np
import tqdm

ageLabels=[(0,6),
           (7,12),
           (13,17),
           (18,25),
           (26,32),
           (33,39),
           (40,45),
           (46,55),
           (56,65),
           (66,1000)]
def getAgeLabel(age):
    age=eval(age)
    if age is None:
        return -1
    else:
        for c,(low,upper) in enumerate(ageLabels):
            if age>=low and age<=upper:
                return c
    return -1

class UTK_Dataset(Dataset):
    def __init__(self,model,filepath):
        super().__init__()
        self._db=[]
        self.model=model

        flist=glob.glob(filepath)
        for filename in tqdm.tqdm(flist):
            id=os.path.basename(filename)
            # [age]_[gender]_[race]_[date&time].jpg
            sps=id.split('_')
            if len(sps)==4:
                age,gender,race,_=sps
            else:
                print(id)
                age, gender, race = sps[0],sps[1],4
            self._db.append((filename,getAgeLabel(age),int(gender),int(race)))

        #统一一下
        agestat=[0]*len(ageLabels)
        genderstat=[0]*2
        racestat=[0]*5
        for p in self._db:
            age,gender,race=p[1:4]
            agestat[age]+=1
            genderstat[int(gender)] += 1
            racestat[int(race)] += 1
        ##show result
        for r,cnt in zip(ageLabels,agestat):
            print('%s:%s'%(str(r),str(cnt)))

        for r,cnt in zip(['M','F'],genderstat):
            print('%s:%s'%(str(r),str(cnt)))
        for r,cnt in zip(['White','Black','Asian','Indian','Others'],racestat):
            print('%s:%s'%(str(r),str(cnt)))

    def __getitem__(self,k):
        filename,age,gender,race=self._db[k]

        I=cv2.imread(filename)
        I = I[:, :, ::-1]
        I=cv2.resize(I,(160,160))
        I = I[None]
        I=np.transpose(I,(0,3,1,2))
        X = prewhiten(I)
        with torch.no_grad():
            X=self.model(X)[0]

        return X,torch.tensor(age),torch.tensor(gender)
    def __len__(self):
        return len(self._db)
if __name__ == '__main__':
    model = InceptionResnetV1(pretrained='casia-webface').eval()
    trainset=UTK_Dataset(model,'/home/zxk/AI/data/UTKFace/*.jpg')

    batch_size=32
    trainloader=DataLoader(trainset,batch_size,shuffle=True,num_workers=4)
    for x in trainloader:
        print(x[0].shape)