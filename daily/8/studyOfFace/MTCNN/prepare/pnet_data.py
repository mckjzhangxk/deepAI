from Configure import WIDER_TRAINSET,WIDER_ANNOTION,PNET_DATASET_PATH,BASE_NUM,POSITIVE_COPYS,NEGATIVE_COPYS,NEG_NUM_FOR_PNET
from utils.roi_utils import IoU,GetRegressBox
import os
import cv2
import numpy as np
import numpy.random as npr

'''
    filename:原图片的路径
    face_coordnte:[],原图片中人脸的所有坐标,(x1,y1,x2,y2),你要验证一下坐标的有效性
    posNum:生成+样本的数量
    negNum:生成-样本的数量
    
    返回一个list,list[i] 是一个dict
        key:label:1->+,-1->part,0->-
           :regbox:对于+,part例,对左上角和右下角的微调
           :coodinate:切割图片的坐标
'''
def _genImage(filename, face_coordnate, posCopys, negCopy, negNum):
    I=cv2.imread(filename)
    H,W,_=I.shape

    face_cood_valid=[]
    for x1,y1,x2,y2 in face_coordnate:
        if(x1+SIZE<x2 and x1>=0 and x2<=W and y1+SIZE<y2 and y1>=0 and y2<=H ):
            face_cood_valid.append((x1,y1,x2,y2))
    if len(face_cood_valid)==0:return []
    ret=[]

    # 生成正样本
    for x1,y1,x2,y2 in face_cood_valid:
        cenx, ceny = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        w, h = x2 - x1, y2 - y1

        #对于一张人脸,生成posCopys个副本
        for n_p in range(posCopys):
            dx=npr.randint(-0.2*w,0.2*w)
            dy=npr.randint(-0.2*h,0.2*h)
            sz=np.random.randint(0.8*min(w,h),1.25*max(w,h))


            nx1,ny1=max(cenx+dx-sz/2,0),max(ceny+dy-sz/2,0)
            nx2,ny2=nx1+sz,ny1+sz

            #防止出边框,或者图片太小
            if nx2>W or ny2>H:continue
            if nx1+SIZE>=nx2 or ny1+SIZE>=ny2:continue

            iou=IoU((nx1,ny1,nx2,ny2),np.array([[x1,y1,x2,y2]]))
            iou=iou[0]
            sample={}
            if iou>0.65:
                sample['label']=1
            elif iou>0.4:
                sample['label'] = -1
            else:continue
            sample['coodinate'] = [nx1, ny1, nx2, ny2]
            sample['regbox'] = GetRegressBox(
                (x1, y1, x2, y2),  # 人脸坐标
                (nx1, ny1, nx2, ny2)  # 截图坐标
            )
            ret.append(sample)


    face_cood_valid = np.array(face_cood_valid)  # (N,4)
    # 生成负样本
    for x1, y1, x2, y2 in face_cood_valid:
        for __ in range(negCopy):
            sz=npr.randint(SIZE,min(W, H) / 2)
            nx1,ny1=max(x1-sz,0),max(y1-sz,0)
            nx2,ny2=nx1+sz,ny1+sz
            if nx2>W or ny2>H:continue
            if nx1+SIZE>=nx2 or ny1+SIZE>=ny2:continue

            iou=IoU((nx1,ny1,nx2,ny2),face_cood_valid)
            iou=np.max(iou)
            if iou<0.3:
                sample = {}
                sample['label']=0
                sample['regbox']=[0,0,0,0]
                sample['coodinate']=[nx1,ny1,nx2,ny2]
                ret.append(sample)
    # 生成负样本

    for _ in range(negNum):
        sz=npr.randint(SIZE,min(W, H) / 2)
        nx1,ny1=npr.randint(0,W-sz),npr.randint(0,H-sz)
        nx2,ny2=nx1+sz,ny1+sz
        if nx1 + SIZE >= nx2 or ny1 + SIZE >= ny2: continue

        iou=IoU((nx1,ny1,nx2,ny2),face_cood_valid)
        iou=np.max(iou)

        if iou<0.3:
            sample = {}
            sample['label']=0
            sample['regbox']=[0,0,0,0]
            sample['coodinate']=[nx1,ny1,nx2,ny2]
            ret.append(sample)
    return ret
'''
往文件f中追加记录
    根据 info的信息
    info.label =1
        追加格式:
        filepath 1 x1 y1 x2 y2
    info.label= -1
        filepath -1 x1 y1 x2 y2
    info.label= 0
        filepath 0
    filepath=PNET_DATASET/{pos|neg|part}/{faceid}.jpg
    x1,y1,x2,y2=info[regbox][0],info[regbox][1],info[regbox][2],info[regbox][3]
    
    然后保存图片
    图片路径:
        {PNET_DATASET}/{pos|neg|part}/{pos|neg|part}.jpg
'''
def _writeAnnationAndImage(info, fs, faceid, orgin_image_path):
    face_output_path=''
    f=None
    imagename=str(faceid)+'.jpg'
    outputline=''


    if info['label'] == 1:
        face_output_path=os.path.join(PNET_DATASET_PATH, 'pos', imagename)
        f=fs[0]
    elif info['label'] == 0:
        face_output_path = os.path.join(PNET_DATASET_PATH, 'neg', imagename)
        f = fs[1]
    elif info['label'] == -1:
        face_output_path = os.path.join(PNET_DATASET_PATH, 'part', imagename)
        f=fs[2]

    outputline = '%s %d %.2f %.2f %.2f %.2f\n' % (face_output_path,info['label'],*info['regbox'])

    f.write(outputline)

    #保存图片
    x1,y1,x2,y2=list(map(int,info['coodinate']))
    I=cv2.imread(orgin_image_path)

    Icrop=I[y1:y2,x1:x2]
    Iresize=cv2.resize(Icrop,(SIZE,SIZE),interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(face_output_path,Iresize)
def _prepareOutDir():
    pos_dir=os.path.join(PNET_DATASET_PATH, 'pos')
    neg_dir=os.path.join(PNET_DATASET_PATH, 'neg')
    part_dir=os.path.join(PNET_DATASET_PATH, 'part')
    if not os.path.exists(PNET_DATASET_PATH):
        os.mkdir(PNET_DATASET_PATH)
    if not os.path.exists(pos_dir):
        os.mkdir(pos_dir)
    if not os.path.exists(neg_dir):
        os.mkdir(neg_dir)
    if not os.path.exists(part_dir):
        os.mkdir(part_dir)
def _summary():
    f_pos=open(os.path.join(PNET_DATASET_PATH, 'pos.txt'), 'r')
    f_neg = open(os.path.join(PNET_DATASET_PATH, 'neg.txt'), 'r')
    f_part = open(os.path.join(PNET_DATASET_PATH, 'part.txt'), 'r')

    pos_num=len(f_pos.readlines())
    neg_num = len(f_neg.readlines())
    part_num = len(f_part.readlines())

    f_pos.close()
    f_neg.close()
    f_part.close()
    print('total positive samples %d' % pos_num)
    print('total negative samples %d' % neg_num)
    print('total part     samples %d' % part_num)

def merge_pnet_dataset(showlog=True):
    f_pos=open(os.path.join(PNET_DATASET_PATH, 'pos.txt'), 'r')
    f_neg = open(os.path.join(PNET_DATASET_PATH, 'neg.txt'), 'r')
    f_part = open(os.path.join(PNET_DATASET_PATH, 'part.txt'), 'r')
    c_pos=  f_pos.readlines()
    c_neg=  f_neg.readlines()
    c_part= f_part.readlines()
    f_pos.close()
    f_neg.close()
    f_part.close()

    pos_num=len(c_pos)
    neg_num = len(c_neg)
    part_num = len(c_part)


    if showlog:
        print('before merge positive samples %d' % pos_num)
        print('before merge negative samples %d' % neg_num)
        print('before merge part     samples %d' % part_num)

    fout=open(os.path.join(PNET_DATASET_PATH, 'PNet.txt'), 'w')

    if pos_num<BASE_NUM:
        pos_num=npr.choice(pos_num,BASE_NUM,True)
    else:
        pos_num = npr.choice(pos_num, BASE_NUM, False)

    if part_num < BASE_NUM:
        part_num = npr.choice(part_num, BASE_NUM, True)
    else:
        part_num = npr.choice(part_num, BASE_NUM, False)

    if neg_num < 2*BASE_NUM:
        neg_num = npr.choice(neg_num, 2*BASE_NUM, True)
    else:
        neg_num = npr.choice(neg_num, 2*BASE_NUM, False)

    for idx in pos_num:
        fout.write(c_pos[idx])
    for idx in neg_num:
        fout.write(c_neg[idx])
    for idx in part_num:
        fout.write(c_part[idx])
    fout.close()

    if showlog:
        print('After merge positive samples %d' % len(pos_num))
        print('After merge negative samples %d' % len(neg_num))
        print('After merge part     samples %d' % len(part_num))

def gen_pnet_data(posCopys, negCopys, negNum):
    _prepareOutDir()

    fs=open(WIDER_ANNOTION,'r')
    lines=fs.readlines()
    cnt=len(lines)
    numOFImages=0
    idx=0

    faceid=0

    f_pos=open(os.path.join(PNET_DATASET_PATH, 'pos.txt'), 'w')
    f_neg = open(os.path.join(PNET_DATASET_PATH, 'neg.txt'), 'w')
    f_part = open(os.path.join(PNET_DATASET_PATH, 'part.txt'), 'w')

    while(idx<cnt):
        name=lines[idx].strip('\n')
        imagepath=os.path.join(WIDER_TRAINSET,name)

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
        samplelist=_genImage(imagepath, face_coordnate, posCopys=posCopys, negCopy=negCopys, negNum=negNum)

        for sm in samplelist:
            faceid += 1
            _writeAnnationAndImage(sm, [f_pos, f_neg, f_part], faceid, imagepath)
        #跳到下一张图片
        idx+=1
        numOFImages+=1
        if numOFImages%2==0:
            print("finish %d/%d"%(numOFImages,12880))
    f_pos.close()
    f_neg.close()
    f_part.close()
    fs.close()

    print('total num of samples %d'%faceid)


if __name__ == '__main__':
    SIZE=12
    # gen_pnet_data(posCopys=POSITIVE_COPYS, negCopys=NEGATIVE_COPYS, negNum=NEG_NUM)
    # _summary()
    merge_pnet_dataset()
