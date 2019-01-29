from Configure import WIDER_TRAIN_ANNOTION,WIDER_TRAINSET,LWF_ANNOTION,LWF_TRAINSET
import os
import numpy as np
import numpy.random as npr
import cv2
from utils.roi_utils import  validRegion,validLandmark,ImageTransform,IoU,GetRegressBox
from utils.common import progess_print

'''
把LFW数据集关于五官的标注进行数据加强后:
图片缩放到SIZE,输出到output_dir/landmark下面
标注输出到输出到output_dir/landmark.txt

'''

def getLFW(SIZE=12,output_dir=None,numOfShift=1):
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
    transutil=ImageTransform()
    fs_anno=open(os.path.join(output_dir,'landmark.txt'),'w')
    outpud_image_dir=os.path.join(output_dir,'landmark')

    def imresample(img, sz):
        if isinstance(sz,int):
            sz=(sz,sz)
        im_data = cv2.resize(img, (sz[1], sz[0]), interpolation=cv2.INTER_AREA)  # @UndefinedVariable
        return im_data


    def choiocebox(box,gtbox):
        def box_too_small(box):
            w, h = box[2] - box[0], box[3] - box[1]
            return max(w, h) < 40
        nx1, ny1, nx2, ny2=box[0],box[1],box[2],box[3]
        if box_too_small([nx1, ny1, nx2, ny2]): return False
        iou = IoU([nx1, ny1, nx2, ny2], gtbox)[0]
        if iou < 0.65: return False
        return True

    def f(I,gtbox,gtlandmark):
        if isinstance(gtbox,list):
            gtbox=np.array(gtbox)
        if isinstance(gtlandmark,list):
            gtlandmark=np.array(gtlandmark)

        H,W,_=I.shape
        '''
        如果标注的区域不合法或者太小的化
        '''
        if not validRegion(gtbox,W,H):
            return np.empty((0,SIZE,SIZE,3)),np.empty((0,10))

        #分别用于保存输出的图片,和输出的landmakk,元素分别是(SIZE,SIZE,3),和(10,)
        image_list,landmark_list=[],[]

        #保存原图
        # image_list.append(imresample(I,(SIZE,SIZE)))
        # landmark_list.append(GetLandMarkPoint(gtbox,gtlandmark))
        for i in range(numOfShift):
            '''
            随机移动,验证切割区域不能太小,不能不能iou过小,不能是非法区域
            '''
            nx1,ny1,nx2,ny2=transutil.shift(gtbox,W,H)
            if not choiocebox([nx1,ny1,nx2,ny2],gtbox):continue


            '''
            转landmark~(0,1)验证有效性
            '''
            landmark_shift=transutil.projectAndNorm(gtlandmark,(nx1, ny1, nx2, ny2))
            if not validLandmark(landmark_shift):continue
            I_shift=I[ny1:ny2, nx1:nx2]
            image_list.append(imresample(I_shift,(SIZE,SIZE)))
            landmark_list.append(landmark_shift)

            '''做镜像处理
            '''
            if npr.choice([0,1])>0:
                I_mirror,landmark_mirror=transutil.flip(
                    I_shift,
                    transutil.projectAndNorm(gtlandmark,[nx1,ny1,nx2,ny2])
                )
                image_list.append(imresample(I_mirror, (SIZE, SIZE)))
                landmark_list.append(landmark_mirror)
            for kk in range(2):
                if npr.choice([0, 1]) > 0:
                    angle=5 if kk==0 else -5
                    I_rotate, landmark_rotate = transutil.rotate(
                        I_shift,
                        transutil.project(gtlandmark,[nx1,ny1,nx2,ny2]),
                        angle
                    )
                    landmark_rotate=transutil.project(landmark_rotate,[nx1,ny1,nx2,ny2],to=False)
                    landmark_rotate=transutil.projectAndNorm(landmark_rotate, (nx1, ny1, nx2, ny2))
                    if not validLandmark(landmark_rotate):continue
                    image_list.append(imresample(I_rotate,SIZE))
                    landmark_list.append(landmark_rotate)

                    I_rotate_mirror, landmark_rotate_mirror=transutil.flip(I_rotate,landmark_rotate)
                    image_list.append(imresample(I_rotate_mirror,SIZE))
                    landmark_list.append(landmark_rotate_mirror)

        return image_list,np.array(landmark_list).reshape((-1,10))

    fs=open(LWF_ANNOTION,'r')
    lines=fs.readlines()
    n_id=0
    for idx,l in enumerate(lines):
        spits=l.strip('\n').split( )
        filepath=os.path.join(LWF_TRAINSET,spits[0].replace('\\','/'))
        assert os.path.exists(filepath),'Image does not exist'

        I=cv2.imread(filepath)
        face_box=np.array([int(x) for x in spits[1:5]])
        face_box=face_box[[0,2,1,3]]
        landmark=np.array([float(x) for x in spits[5:]])

        imgs,aug_landmarks=f(I,face_box,landmark)
        for img,x in zip(imgs,aug_landmarks):
            out_file=os.path.join(outpud_image_dir,'%d.jpg'%n_id)
            fs_anno.write('%s %d %d %d %d %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n'%
                     (out_file,-2,0,0,0,0,
                      x[0],x[1],
                      x[2], x[3],
                      x[4], x[5],
                      x[6], x[7],
                      x[8], x[9])
                     )
            cv2.imwrite(out_file,img)
            n_id+=1
        if idx%20==0:
            progess_print('finish landmark convert %d/%d'%(idx+1,len(lines)))
    print('total %d examples'%n_id)
    fs.close()
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

    fs = open(WIDER_TRAIN_ANNOTION, 'r')
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
    fs = open(WIDER_TRAIN_ANNOTION, 'r')
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


def genImage(filename, face_coordnate, posCopys, negCopy, negNum,SIZE):
    tsutils = ImageTransform()
    I = cv2.imread(filename)
    H, W, _ = I.shape

    face_cood_valid = []
    for x1, y1, x2, y2 in face_coordnate:
        if (x1 + SIZE < x2 and x1 >= 0 and x2 <= W and y1 + SIZE < y2 and y1 >= 0 and y2 <= H):
            face_cood_valid.append((x1, y1, x2, y2))
    if len(face_cood_valid) == 0: return []
    ret = []

    # 生成正样本
    for x1, y1, x2, y2 in face_cood_valid:
        if not validRegion([x1, y1, x2, y2], W, H): continue
        # 对于一张人脸,生成posCopys个副本
        for n_p in range(posCopys):
            nx1, ny1, nx2, ny2 = tsutils.shift([x1, y1, x2, y2], W, H)
            if nx1 < 0:
                continue
            iou = IoU((nx1, ny1, nx2, ny2), np.array([[x1, y1, x2, y2]]))
            iou = iou[0]
            sample = {}
            if iou > 0.65:
                sample['label'] = 1
            elif iou > 0.4:
                sample['label'] = -1
            else:
                continue
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
            sz = npr.randint(SIZE, min(W, H) / 2)
            nx1, ny1 = max(x1 - sz, 0), max(y1 - sz, 0)
            nx2, ny2 = nx1 + sz, ny1 + sz
            if nx2 > W or ny2 > H: continue
            if nx1 + SIZE >= nx2 or ny1 + SIZE >= ny2: continue

            iou = IoU((nx1, ny1, nx2, ny2), face_cood_valid)
            iou = np.max(iou)
            if iou < 0.3:
                sample = {}
                sample['label'] = 0
                sample['regbox'] = [0, 0, 0, 0]
                sample['coodinate'] = [nx1, ny1, nx2, ny2]
                ret.append(sample)
    # 生成负样本

    for _ in range(negNum):
        sz = npr.randint(SIZE, min(W, H) / 2)
        nx1, ny1 = npr.randint(0, W - sz), npr.randint(0, H - sz)
        nx2, ny2 = nx1 + sz, ny1 + sz
        if nx1 + SIZE >= nx2 or ny1 + SIZE >= ny2: continue

        iou = IoU((nx1, ny1, nx2, ny2), face_cood_valid)
        iou = np.max(iou)

        if iou < 0.3:
            sample = {}
            sample['label'] = 0
            sample['regbox'] = [0, 0, 0, 0]
            sample['coodinate'] = [nx1, ny1, nx2, ny2]
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
    filepath=OUTPATH/{pos|neg|part}/{faceid}.jpg
    x1,y1,x2,y2=info[regbox][0],info[regbox][1],info[regbox][2],info[regbox][3]

    然后保存图片
    图片路径:
        {OUTPATH}/{pos|neg|part}/faceid.jpg
'''


def writeAnnationAndImage(info, fs, faceid, orgin_image_path, OUTPATH, SIZE):
    face_output_path = ''
    f = None
    imagename = str(faceid) + '.jpg'


    if info['label'] == 1:
        face_output_path = os.path.join(OUTPATH, 'pos', imagename)
        f = fs[0]
    elif info['label'] == 0:
        face_output_path = os.path.join(OUTPATH, 'neg', imagename)
        f = fs[1]
    elif info['label'] == -1:
        face_output_path = os.path.join(OUTPATH, 'part', imagename)
        f = fs[2]

    outputline = '%s %d %.2f %.2f %.2f %.2f 0 0 0 0 0 0 0 0 0 0\n' % (face_output_path, info['label'], *info['regbox'])
    f.write(outputline)

    # 保存图片
    x1, y1, x2, y2 = list(map(int, info['coodinate']))
    I = cv2.imread(orgin_image_path)

    Icrop = I[y1:y2, x1:x2]
    Iresize = cv2.resize(Icrop, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(face_output_path, Iresize)

    if info['label'] == 1:
        imagename = str(faceid + 1) + '.jpg'
        face_output_path = os.path.join(OUTPATH, 'pos', imagename)
        rx1, ry1, rx2, ry2 = info['regbox']
        outputline = '%s %d %.2f %.2f %.2f %.2f 0 0 0 0 0 0 0 0 0 0\n' % (face_output_path, 1, -rx2, ry1, -rx1, ry2)
        f.write(outputline)
        cv2.flip(Iresize, 1, Iresize)
        cv2.imwrite(face_output_path, Iresize)
        return 2
    else:
        return 1

def prepareOutDir(DATASET_PATH):
    if not os.path.exists(DATASET_PATH):
        os.mkdir(DATASET_PATH)
    sx=['pos','neg','part','landmark']
    for x in sx:
        path=os.path.join(DATASET_PATH, x)
        if not os.path.exists(path):
            os.mkdir(path)