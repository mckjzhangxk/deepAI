
from  data.Conf import DataConf
import numpy as np
import os

def read_data(datafile,logfs=None):
    pos=[]
    neg=[]

    with open(datafile,'r') as fs:
        lines=fs.readlines()
    for l in lines:
        if l.startswith('1,'):
            pos.append(l)
        else:
            neg.append(l)

    if logfs:
        filename=os.path.basename(datafile)
        logfs.write('%s\n\t#pos:%d\n\t#neg:%d\n'%(filename,len(pos),len(neg)))
    return pos,neg
def write2File(pos,pos_idx,neg,neg_idx,outputpath):
    pos_out=[pos[i] for i in pos_idx]
    neg_out=[neg[i] for i in neg_idx]

    out=pos_out+neg_out
    np.random.shuffle(out)
    with open(outputpath,'w') as fs:
        for l in out:
            fs.write(l)
if __name__ == '__main__':
    DATASET='run_train'


    sourcepath=DataConf.TrainDataSource
    fnames=sorted(os.listdir(sourcepath))
    pos, neg = [], []
    with open(DataConf.Dataset_Log,'w') as logfs:

        for fname in fnames:
            _pos, _neg=read_data(os.path.join(sourcepath,fname),logfs)
            pos=pos+_pos
            neg=neg+_neg
    print('#pos %d,#neg %d'%(len(pos),len(neg)))

    if len(pos)<DataConf.BASE_NUM:
        pos_idx=np.random.choice(len(pos),DataConf.BASE_NUM,True)
    else:
        pos_idx = np.random.choice(len(pos), DataConf.BASE_NUM, False)

    if len(neg)<DataConf.BASE_NUM:
        neg_idx=np.random.choice(len(neg),DataConf.BASE_NUM,True)
    else:
        neg_idx = np.random.choice(len(neg), DataConf.BASE_NUM, False)
    if DATASET=='run_train':
        write2File(pos, pos_idx, neg, neg_idx, DataConf.TrainFile)
    else:
        write2File(pos,pos_idx,neg,neg_idx,DataConf.DevFile)