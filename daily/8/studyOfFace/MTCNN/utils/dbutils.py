from Configure import WIDER_ANNOTION,WIDER_TRAINSET,LWF_ANNOTION,LWF_TRAINSET
import os
import numpy as np
import cv2
from utils.roi_utils import  GetLandMarkPoint


'''
把LFW数据集关于五官的标注进行数据加强后:
图片缩放到SIZE,输出到output_dir/landmark下面
标注输出到输出到output_dir/landmark.txt

'''

def getLFW(SIZE=12,output_dir=None):
    '''
    I表示一张LFW图片,
    gtbox:标注的face box (4,)
    gtlandmark:标注的五官(10,)
    
    图像加强算法
        1.镜像
        2.旋转
        3.平移
        
    :param SIZE: 缩放尺寸
    :return: images:(N,SIZE,SIZE,3) landmarks:(N,10)
    '''
    fs_anno=open(os.path.join(output_dir,'landmark.txt'),'w')
    outpud_image_dir=os.path.join(output_dir,'landmark')

    def f(I,gtbox,gtlandmark):
        if isinstance(gtbox,list):
            gtbox=np.array(gtbox)
        if isinstance(gtlandmark,list):
            gtlandmark=np.array(gtlandmark)

        I=np.expand_dims(I,0)
        gtlandmark = np.expand_dims(gtlandmark, 0)
        return I,gtlandmark

    fs=open(LWF_ANNOTION,'r')
    lines=fs.readlines()
    n_id=0
    for l in lines:
        spits=l.strip('\n').split( )
        filepath=os.path.join(LWF_TRAINSET,spits[0].replace('\\','/'))
        assert os.path.exists(filepath),'Image does not exist'

        I=cv2.imread(filepath)
        face_box=np.array([int(x) for x in spits[1:5]])
        face_box=face_box[[0,2,1,3]]
        landmark=np.array([float(x) for x in spits[5:]])

        imgs,aug_landmarks=f(I,face_box,landmark)
        for img,x in zip(imgs,aug_landmarks):
            out_file=os.path.join(outpud_image_dir+'%d.jpg'%n_id)

            fs_anno.write('%s %d %d %d %d %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n'%
                     (out_file,-2,0,0,0,0,
                      x[0],x[1],
                      x[2], x[3],
                      x[4], x[5],
                      x[6], x[7],
                      x[8], x[9])
                     )
            n_id+=1
    fs.close()
if __name__ == '__main__':
    getLFW(SIZE=12,output_dir='/home/zhangxk/AIProject/MTCNN_TRAIN/rnet/dataset')
'''
返回一个dict
    key:图片名:
    value:np.array([N',4])标注的人脸
    这里过滤了过小人脸(size<12)
    
    total num of images 12880
    total num of faces 94484
'''
def get_WIDER_Set():
    ret=dict()

    fs = open(WIDER_ANNOTION, 'r')
    lines=fs.readlines()
    cnt=len(lines)
    numOFImages=0
    numOFFace=0
    SIZE=12
    idx=0


    while (idx < cnt):
        name = lines[idx].strip('\n')
        imagepath = os.path.join(WIDER_TRAINSET, name)
        I = cv2.imread(imagepath)
        H, W, _ = I.shape

        # 获得人脸数目
        idx += 1
        facenum = int(lines[idx].strip('\n'))

        face_coordnate=[]
        for n in range(facenum):
            idx+=1
            splits = lines[idx].split(' ')
            x1, y1, w, h = float(splits[0]), float(splits[1]), float(splits[2]), float(splits[3])
            x2, y2 = x1 + w, y1 + h
            if (x1 + SIZE < x2 and x1 >= 0 and x2 <= W and y1 + SIZE < y2 and y1 >= 0 and y2 <= H):
                face_coordnate.append((x1,y1,x2,y2))
        numOFImages += 1
        numOFFace+=len(face_coordnate)
        #跳到下一张图片
        idx+=1
        ret[imagepath]=np.array(face_coordnate)
    fs.close()

    print('total num of images %d' % numOFImages)
    print('total num of faces %d' % numOFFace)
    return ret
def get_WIDER_Set_ImagePath():
    ret=[]
    fs = open(WIDER_ANNOTION, 'r')
    lines = fs.readlines()
    cnt = len(lines)

    idx = 0

    while (idx < cnt):
        name = lines[idx].strip('\n')
        imagepath = os.path.join(WIDER_TRAINSET, name)

        # 获得人脸数目
        idx += 1
        facenum = int(lines[idx].strip('\n'))
        idx =idx+ facenum+1
        if os.path.exists(imagepath):
            ret.append(imagepath)
    fs.close()
    return ret

'''
统计pos+neg+part 样本数量
'''
def get_example_nums(basedir,fnames=None):
    if fnames is None:
        fnames=['pos.txt','neg.txt','part.txt']
    if not isinstance(fnames,list):
        fnames=[fnames]

    cnt=0
    for i,fname in enumerate(fnames):
        file_dir=os.path.join(basedir,fname)
        assert os.path.exists(file_dir), '%s not exist' % file_dir
        fs=open(file_dir,'r')
        examples=len(fs.readlines())
        print('%s have %d exmaples'%(fname,examples))
        cnt+=examples
    print('Total have %d exmaples' % (cnt))
    return cnt