from utils.dbutils import genImage,writeAnnationAndImage,prepareOutDir
from Configure import PNET_DATASET_VALID_PATH,WIDER_VALSET,WIDER_VAL_ANNOTION
import numpy.random as npr
import os
from utils.common import progess_print
from utils.tf_utils import cvtTxt2TF

def gen_pnet_valid_data(posCopys, negCopys, negNum):
    prepareOutDir(PNET_DATASET_VALID_PATH)

    fs=open(WIDER_VAL_ANNOTION, 'r')
    lines=fs.readlines()
    cnt=len(lines)
    numOFImages=0
    idx=0

    faceid=0
    f_pos=open(os.path.join(PNET_DATASET_VALID_PATH, 'pos.txt'), 'w')
    f_neg = open(os.path.join(PNET_DATASET_VALID_PATH, 'neg.txt'), 'w')

    while(idx<cnt):
        name=lines[idx].strip('\n')
        imagepath=os.path.join(WIDER_VALSET,name)

        assert os.path.exists(imagepath) ,'file does not exist'
        #获得人脸数目
        idx+=1
        facenum=int(lines[idx].strip('\n'))

        #获得人脸坐标
        face_coordnate=[]
        for n in range(facenum):
            idx+=1
            splits = lines[idx].split(' ')
            x1, y1, w, h = float(splits[0]), float(splits[1]), float(splits[2]), float(splits[3])
            x2, y2 = x1 + w, y1 + h
            face_coordnate.append((x1,y1,x2,y2))
        '''
        给一张原图,图上所有人脸的坐标,返回negNum个非人脸,对于每个人脸,
        生成 posCopys张人脸副本,这些副本就会有了regbox,也就是人脸坐标的修正!
        这里提供了一张图片可生成的全部样本!
        '''
        samplelist=genImage(imagepath, face_coordnate, posCopys=posCopys, negCopy=negCopys, negNum=negNum,SIZE=SIZE)

        for sm in samplelist:
            if sm['label']==-1:continue
            _cnt=writeAnnationAndImage(sm, [f_pos, f_neg], faceid, imagepath,PNET_DATASET_VALID_PATH,SIZE)
            faceid += _cnt

        #跳到下一张图片
        idx+=1
        numOFImages+=1
        if numOFImages%2==0:
            progess_print("finish %d/%d"%(numOFImages,12880))
    f_pos.close()
    f_neg.close()
    fs.close()

    print('total num of samples %d'%faceid)

def merge_pnet_dataset(showlog=True):
    f_pos=open(os.path.join(PNET_DATASET_VALID_PATH, 'pos.txt'), 'r')
    f_neg = open(os.path.join(PNET_DATASET_VALID_PATH, 'neg.txt'), 'r')


    c_pos=  f_pos.readlines()
    c_neg=  f_neg.readlines()

    f_pos.close()
    f_neg.close()

    pos_num=len(c_pos)
    neg_num = len(c_neg)

    BASE_NUM=pos_num

    if showlog:
        print('before merge positive samples %d' % pos_num)
        print('before merge negative samples %d' % neg_num)


    fout=open(os.path.join(PNET_DATASET_VALID_PATH, 'PNet.txt'), 'w')

    pos_num = npr.choice(pos_num, BASE_NUM, False)
    if neg_num < BASE_NUM:
        neg_num = npr.choice(neg_num, BASE_NUM, True)
    else:
        neg_num = npr.choice(neg_num, BASE_NUM, False)


    for idx in pos_num:
        fout.write(c_pos[idx])
    for idx in neg_num:
        fout.write(c_neg[idx])

    fout.close()

    if showlog:
        print('After merge positive samples %d' % len(pos_num))
        print('After merge negative samples %d' % len(neg_num))


if __name__ == '__main__':
    SIZE=12
    gen_pnet_valid_data(posCopys=1,negCopys=1,negNum=0)
    merge_pnet_dataset()
    cvtTxt2TF(PNET_DATASET_VALID_PATH, 'PNet.txt', 'PNet_shuffle', True)