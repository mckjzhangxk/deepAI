
from  data.Conf import DataConf
import numpy as np


def read_data(datafile):
    pos=[]
    neg=[]

    with open(datafile,'r') as fs:
        lines=fs.readlines()
    for l in lines:
        if l.startswith('1,'):
            pos.append(l)
        else:
            neg.append(l)
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
    pos, neg=read_data(DataConf.data_path)

    # print('# pos is %d'%(len(pos)))
    # print('# neg is %d' % (len(neg)))

    if len(pos)<DataConf.BASE_NUM:
        pos_idx=np.random.choice(len(pos),DataConf.BASE_NUM,True)
    else:
        pos_idx = np.random.choice(len(pos), DataConf.BASE_NUM, False)

    if len(neg)<DataConf.BASE_NUM:
        neg_idx=np.random.choice(len(neg),DataConf.BASE_NUM,True)
    else:
        neg_idx = np.random.choice(len(neg), DataConf.BASE_NUM, False)

    write2File(pos, pos_idx, neg, neg_idx, DataConf.TrainFile)
    # write2File(pos,pos_idx,neg,neg_idx,DataConf.DevFile)