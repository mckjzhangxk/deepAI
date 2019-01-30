from Configure import WIDER_TRAINSET,WIDER_TRAIN_ANNOTION,PNET_DATASET_PATH,BASE_NUM,POSITIVE_COPYS,NEGATIVE_COPYS,NEG_NUM_FOR_PNET,LWF_SHIFT
from utils.dbutils import getLFW,genImage,writeAnnationAndImage,prepareOutDir
import os
import numpy.random as npr
import numpy as np
from utils.common import progess_print




def _summary():
    f_pos=open(os.path.join(PNET_DATASET_PATH, 'pos.txt'), 'r')
    f_neg = open(os.path.join(PNET_DATASET_PATH, 'neg.txt'), 'r')
    f_part = open(os.path.join(PNET_DATASET_PATH, 'part.txt'), 'r')
    f_landmark = open(os.path.join(PNET_DATASET_PATH, 'landmark.txt'), 'r')

    pos_num=len(f_pos.readlines())
    neg_num = len(f_neg.readlines())
    part_num = len(f_part.readlines())
    landmark_num=len(f_landmark.readlines())

    f_pos.close()
    f_neg.close()
    f_part.close()
    f_landmark.close()


    print('total positive samples %d' % pos_num)
    print('total negative samples %d' % neg_num)
    print('total part     samples %d' % part_num)
    print('total landmark samples %d' % landmark_num)

def merge_pnet_dataset(showlog=True):
    f_pos=open(os.path.join(PNET_DATASET_PATH, 'pos.txt'), 'r')
    f_neg = open(os.path.join(PNET_DATASET_PATH, 'neg.txt'), 'r')
    f_part = open(os.path.join(PNET_DATASET_PATH, 'part.txt'), 'r')
    f_landmark = open(os.path.join(PNET_DATASET_PATH, 'landmark.txt'), 'r')

    c_pos=  f_pos.readlines()
    c_neg=  f_neg.readlines()
    c_part= f_part.readlines()
    c_landmark = f_landmark.readlines()

    f_pos.close()
    f_neg.close()
    f_part.close()
    f_landmark.close()

    pos_num=len(c_pos)
    neg_num = len(c_neg)
    part_num = len(c_part)
    landmark_num=len(c_landmark)

    if showlog:
        print('before merge positive samples %d' % pos_num)
        print('before merge negative samples %d' % neg_num)
        print('before merge part     samples %d' % part_num)
        print('before merge landmark samples %d' % landmark_num)

    fout=open(os.path.join(PNET_DATASET_PATH, 'PNet.txt'), 'w')

    if pos_num<BASE_NUM:
        pos_num=npr.choice(pos_num,BASE_NUM,True)
    else:
        pos_num = npr.choice(pos_num, BASE_NUM, False)

    if part_num < BASE_NUM:
        part_num = npr.choice(part_num, BASE_NUM, True)
    else:
        part_num = npr.choice(part_num, BASE_NUM, False)

    if neg_num < 3*BASE_NUM:
        neg_num = npr.choice(neg_num, 3*BASE_NUM, True)
    else:
        neg_num = npr.choice(neg_num, 3*BASE_NUM, False)

    if landmark_num<BASE_NUM:
        landmark_num=npr.choice(landmark_num,BASE_NUM,True)
    else:
        landmark_num = npr.choice(landmark_num, BASE_NUM, False)

    for idx in pos_num:
        fout.write(c_pos[idx])
    for idx in neg_num:
        fout.write(c_neg[idx])
    for idx in part_num:
        fout.write(c_part[idx])
    for idx in landmark_num:
        fout.write(c_landmark[idx])

    fout.close()

    if showlog:
        print('After merge positive samples %d' % len(pos_num))
        print('After merge negative samples %d' % len(neg_num))
        print('After merge part     samples %d' % len(part_num))
        print('After merge landmark samples %d' % len(landmark_num))
def gen_pnet_data(posCopys, negCopys, negNum):
    prepareOutDir(PNET_DATASET_PATH)

    fs=open(WIDER_TRAIN_ANNOTION, 'r')
    lines=fs.readlines()
    cnt=len(lines)
    idx=0
    faceid=0
    f_pos=open(os.path.join(PNET_DATASET_PATH, 'pos.txt'), 'w')
    f_neg = open(os.path.join(PNET_DATASET_PATH, 'neg.txt'), 'w')
    f_part = open(os.path.join(PNET_DATASET_PATH, 'part.txt'), 'w')

    while(idx<cnt):
        line=lines[idx].strip('\n').strip(' ')
        sps=line.split(' ')
        name=sps[0]+'.jpg'
        imagepath=os.path.join(WIDER_TRAINSET,name)

        print(imagepath)
        assert os.path.exists(imagepath) ,'file does not exist'


        #获得人脸坐标
        face_coordnate=np.array([float(x) for x in sps[1:]]).reshape(-1, 4)

        '''
        给一张原图,图上所有人脸的坐标,返回negNum个非人脸,对于每个人脸,
        生成 posCopys张人脸副本,这些副本就会有了regbox,也就是人脸坐标的修正!
        这里提供了一张图片可生成的全部样本!
        '''
        samplelist=genImage(imagepath, face_coordnate, posCopys=posCopys, negCopy=negCopys, negNum=negNum,SIZE=SIZE)

        for sm in samplelist:
            _cnt=writeAnnationAndImage(sm, [f_pos, f_neg, f_part], faceid, imagepath,PNET_DATASET_PATH,SIZE)
            faceid += _cnt
        #跳到下一张图片
        idx+=1

        if idx%2==0:
            progess_print("finish %d/%d"%(idx,cnt))
    f_pos.close()
    f_neg.close()
    f_part.close()
    fs.close()

    print('total num of samples %d'%faceid)


if __name__ == '__main__':
    SIZE=12
    gen_pnet_data(posCopys=POSITIVE_COPYS, negCopys=NEGATIVE_COPYS, negNum=NEG_NUM_FOR_PNET)
    getLFW(SIZE,output_dir=PNET_DATASET_PATH, numOfShift=LWF_SHIFT)
    # _summary()
    merge_pnet_dataset()
