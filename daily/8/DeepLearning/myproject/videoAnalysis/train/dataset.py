import os
import glob
from torch.utils.data import DataLoader,Dataset
import torch
from facenet_pytorch import InceptionResnetV1
from utils.imageUtils import prewhiten

ageLabels=[(0,3),(4,6),(8,13),(12,22),(25,32),(35,45),(48,55),(60,100)]

def getAgeLabel(age):
    age=eval(age)
    if age is None:
        return -1
    else:
        age=age if isinstance(age,int) else (age[0]+age[1])/2
        for c,(low,upper) in enumerate(ageLabels):
            if age>=low and age<=upper:
                return c
    return -1
def getGenderLabel(gender):
    if gender=='' or gender=='u':
        return -1
    if gender=='m':
        return 0
    if gender=='f':
        return 1
def readDataLabel(filepath,imgprefix,target):
    with open(filepath,'r') as fs:
        lines=fs.readlines()

    arr=[]
    for line in lines[1:]:
        record=line.strip().split('\t')
        

    
        orgin=record[0]
        filename=record[1]
        faceid=record[2]
        age=getAgeLabel(record[3])
        gender=getGenderLabel(record[4])
        
        fname=os.path.join(imgprefix,orgin,'coarse_tilt_aligned_face.'+faceid+'.'+filename)

        assert os.path.exists(fname),'%s,not exist'%(fname)      
        if age==-1 and gender==-1:continue
        s='\t'.join([fname,str(age),str(gender)])
        arr.append(s)
    with open(target,'w') as fs:
        for s in arr:
            fs.write(s+'\n')


class Gender_Dataset(Dataset):
    def __init__(self,model,filepath):
        super().__init__()
        self._db=[]
        self.model=model

        with open(filepath) as fs:
            lines=fs.readlines()
        for l in lines:
            s=l.split('\t')
            filename=s[0]
            age=int(s[1])
            gender=int(s[2])
            self._db.append((filename,age,gender))
    def __getitem__(self,k):
        filename,age,gender=self._db[k]

        I=cv2.imread(filename)
        I = I[:, :, ::-1]
        I=cv2.resize(I,(160,160))
        I = I[None]
        I=np.transpose(I,(0,3,1,2))
        X = prewhiten(I)
        return X,torch.tensor(age),torch.tensor(gender)
    def __len__(self):
        return len(self._db)
# readDataLabel(
#     filepath='data/Adience/fold_0_data.txt',
#     imgprefix='/home/zhangxk/AIProject/数据集与模型/faces',
#     target='train.txt')

if __name__ == "__main__":
    model=InceptionResnetV1(pretrained='casia-webface').eval()

    trainset=Gender_Dataset(model,'train.txt')
    batch_size=64
    trainloader=DataLoader(trainset,batch_size,shuffle=True,num_workers=4)
    for x in trainloader:
        print(x[0].shape)