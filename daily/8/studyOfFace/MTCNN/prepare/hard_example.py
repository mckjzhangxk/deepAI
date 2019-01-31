import os
import cv2
import numpy as np
import tensorflow as tf
import pickle
from Configure import RNET_DATASET_PATH, THRESHOLD, NMS_DEFAULT, SCALE, FACE_MIN_SIZE, NEG_NUM_FOR_RNET,DETECT_EPOCHS,ONET_DATASET_PATH,LWF_SHIFT
from detect import cutImage
from detect.Detector import Detector_tf as Detector

from utils.dbutils import get_WIDER_Set, get_WIDER_Set_ImagePath,getLFW
from utils.roi_utils import IoU, GetRegressBox
from utils.common import progess_print

def _prepareOutDir(dataset_path):
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    sx = ['pos', 'neg', 'part', 'landmark']
    for x in sx:
        path=os.path.join(dataset_path, x)
        if not os.path.exists(path):
            os.mkdir(path)
#第一步,使用PNET,对数据集图片生成候选box,map<imagepath,total_box>  predict
def step1(detector,display_every=2):

    imlist=get_WIDER_Set_ImagePath()
    tmp_db={}

    for idx,filename in enumerate(imlist):
        total_box=detector.detect_face(filename)
        if len(total_box)>0:
            tmp_db[filename]=total_box[:,0:4] #返回的是9维的,只要前4
        if idx%display_every==0:
            progess_print('step1,detection task,finish %d/%d'%(idx+1,len(imlist)))
    save_path=os.path.join(RNET_DATASET_PATH,'detections.pkl')
    with open(save_path,'wb') as f:
        pickle.dump(tmp_db,f,1)
    return tmp_db
#第二部,调用get_WIDER_Set,获得标注集合,map<imagepath,total_box> groudtrue
def step2():
    return get_WIDER_Set()


'''
#第三部遍历dets的每个元素,
与groudtrues进行对比筛选,IOU<0.3 -,04<IOU<0.65part,>0.65 +

output_dir下面会输出
    pos.txt/neg.txt/part.txt      //标注文件
    pos/neg/part                  //图片文件
IMG_SIZE:输出pos/neg/part 图片的尺寸
    24,48
'''
def step3(dets, gts, IMG_SIZE, output_dir,display_every=100):
    # 三个标注文件
    pos_file = open(os.path.join(output_dir, 'pos.txt'), 'w')
    neg_file = open(os.path.join(output_dir, 'neg.txt'), 'w')
    part_file = open(os.path.join(output_dir, 'part.txt'), 'w')
    #三个样本图片目录
    pos_dir=os.path.join(output_dir, 'pos')
    neg_dir=os.path.join(output_dir, 'neg')
    part_dir=os.path.join(output_dir, 'part')

    # 正,负,part样本的id(名称)
    n_idx = 0
    p_idx = 0
    d_idx = 0
    image_done = 0

    '''
    im_path 是图片路径
    det_box:是MTCNN的人脸位置 对im_path的检测结果[N',4]
    '''
    for im_path,det_box in dets.items():
        if image_done % display_every == 0:
            progess_print("step3,generate new image task,%d/%d images done" % (image_done,len(dets)))
        image_done += 1
        #标注文件如果有这个,这一步是必要的,因为忽略了标注文件中过小的人脸,可能有的原文件标注不再gts里
        if im_path in gts:
            img = cv2.imread(im_path)
            neg_num = 0
            gts_box=gts[im_path] #标注的针对本图的人脸
            #一个det_box与所有gt_box比较,决定是否通过!
            for box in det_box:
                x1, y1, x2, y2=box.astype(int)
                width = x2-x1
                height = y2-y1

                # 忽略过小,超边框的box
                if min(width,height) < 20 or x1 < 0 or y1 < 0 or x2 > img.shape[1]  or y2 >img.shape[0] :
                    continue

                # 检测出来的box与标注box的IOU
                iou = IoU(box, gts_box)
                iou_max=np.max(iou)
                #给出原图,剪切区域,生成目标图片大小,就能得到目标图片,注意这里是批处理,所以加[0]和[]
                resized_im=cutImage(img, np.array([box]), IMG_SIZE)[0]
                #iou<0.3判定是负样本,针对一张图片不能超过 NEG_NUM_FOR_RNET=60张负样本
                if iou_max < 0.3 and neg_num < NEG_NUM_FOR_RNET:
                    # save the examples
                    save_file = os.path.join(neg_dir, "%s.jpg" % n_idx)
                    neg_file.write(save_file + ' 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n')
                    cv2.imwrite(save_file, resized_im)
                    n_idx += 1
                    neg_num += 1
                else:
                    #检测box 与标注最 接近人脸索引
                    idx = np.argmax(iou)
                    offset_x1, offset_y1, offset_x2, offset_y2=GetRegressBox(gts_box[idx],box)
                    #>0.65,保存为正样本
                    if iou_max >= 0.65:
                        save_file = os.path.join(pos_dir, "%s.jpg" % p_idx)
                        pos_file.write(save_file + ' 1 %.2f %.2f %.2f %.2f 0 0 0 0 0 0 0 0 0 0\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                        cv2.imwrite(save_file, resized_im)
                        p_idx += 1
                    # 0.4,0.65之间,保存为part样本
                    elif iou_max >= 0.4:
                        save_file = os.path.join(part_dir, "%s.jpg" % d_idx)
                        part_file.write(save_file + ' -1 %.2f %.2f %.2f %.2f 0 0 0 0 0 0 0 0 0 0\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                        cv2.imwrite(save_file, resized_im)
                        d_idx += 1

    neg_file.close()
    pos_file.close()
    part_file.close()

    print()
    print('total positive samples %d' % p_idx)
    print('total negative samples %d' % n_idx)
    print('total part     samples %d' % d_idx)

detector=None
def gen_hard_example(net):
    sess = tf.Session()
    model_path = []
    if net=='RNet':
        target_dir=RNET_DATASET_PATH
        IMSIZE=24
        from train_model.solver.pnet_solver import MODEL_CHECKPOINT_DIR as PNET_MODEL_PATH
        model_path.append(os.path.join(PNET_MODEL_PATH, 'PNet-%d' % DETECT_EPOCHS[0]))
    if net=='ONet':
        target_dir=ONET_DATASET_PATH
        IMSIZE=48
        from train_model.solver.pnet_solver import MODEL_CHECKPOINT_DIR as PNET_MODEL_PATH
        model_path.append(os.path.join(PNET_MODEL_PATH, 'PNet-%d' % DETECT_EPOCHS[0]))
        from train_model.solver.rnet_solver import MODEL_CHECKPOINT_DIR as RNET_MODEL_PATH
        model_path.append(os.path.join(RNET_MODEL_PATH, 'RNet-%d' % DETECT_EPOCHS[1]))
    detector = Detector(sess,
                        minsize=FACE_MIN_SIZE,
                        scaleFactor=SCALE,
                        nms=NMS_DEFAULT,
                        threshold=THRESHOLD,
                        model_path=model_path
                        )
    _prepareOutDir(target_dir)
    dets=step1(detector)
    gts=step2()
    step3(dets,gts,IMSIZE,target_dir)
    getLFW(IMSIZE,target_dir,LWF_SHIFT)
