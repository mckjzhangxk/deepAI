'''

公共区域,数据集路径
'''
#TRAINSET
from os.path import join
BASE_DIR='/home/zhangxk/AIProject'
WIDER_TRAINSET=join(BASE_DIR,'WIDER_train/images')
WIDER_TRAIN_ANNOTION=join(BASE_DIR,'WIDER_train/wider_face_train.txt')

WIDER_VALSET=join(BASE_DIR,'WIDER_val/images')
WIDER_VAL_ANNOTION= join(BASE_DIR,'WIDER_val/wider_face_val_bbx_gt.txt')

#LWF用于landmark
LWF_TRAINSET=join(BASE_DIR,'lfw')
LWF_ANNOTION=join(BASE_DIR,'lfw/trainImageList.txt')
LWF_SHIFT=10



FACE_MIN_SIZE=50 #最小人脸尺寸
SCALE=0.79    #人脸金字塔,每次缩放比例
# DETECT_EPOCHS=[3,4,5]  #生成hardexample时候,P,R,O分别使用第EPOCH?个训练的模型
DETECT_EPOCHS=[22,4,2]
#PNET使用2次nms,RNET,ONET各使用一次
NMS_DEFAULT=[0.5,0.7,0.6,0.6]
THRESHOLD=[0.3, 0.1, 0.4]
L2_FACTOR=5e-4
'''

PNET 网络区
'''
#训练P网络数据的基数
BASE_NUM=200000

#PNET数据处理参数,
PNET_DATASET_PATH= join(BASE_DIR,'MTCNN_TRAIN/pnet/dataset')
PNET_DATASET_VALID_PATH=join(BASE_DIR,'MTCNN_TRAIN/pnet/dataset_valid')
POSITIVE_COPYS=10
NEGATIVE_COPYS=5
NEG_NUM_FOR_PNET=50

'''
RNET 网络区

'''
#RNET数据处理
RNET_DATASET_PATH=join(BASE_DIR,'MTCNN_TRAIN/rnet/dataset')
RNET_DATASET_VALID_PATH=join(BASE_DIR,'MTCNN_TRAIN/rnet/dataset_valid')
#一张图片生成最多NEG_NUM_FOR_RNET张人脸
NEG_NUM_FOR_RNET=60
'''

'''
#RNET数据处理
ONET_DATASET_PATH=join(BASE_DIR,'MTCNN_TRAIN/onet/dataset')
ONET_DATASET_VALID_PATH=join(BASE_DIR,'MTCNN_TRAIN/onet/dataset_valid')
#一张图片生成最多NEG_NUM_FOR_RNET张人脸
NEG_NUM_FOR_ONET=60