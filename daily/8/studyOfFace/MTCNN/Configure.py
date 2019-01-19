'''

公共区域,数据集路径
'''
#TRAINSET
WIDER_TRAINSET='/home/zhangxk/AIProject/WIDER_train/images'
WIDER_ANNOTION='/home/zhangxk/AIProject/WIDER_train/wider_face_train_bbx_gt_test.txt'
FACE_MIN_SIZE=50 #最小人脸尺寸
SCALE=0.79    #人脸金字塔,每次缩放比例
'''

PNET 网络区
'''
#训练P网络数据的基数
BASE_NUM=200

#PNET数据处理参数,
PNET_DATASET_PATH= '/home/zhangxk/AIProject/MTCNN_TRAIN/pnet/dataset'
POSITIVE_COPYS=10
NEGATIVE_COPYS=5
NEG_NUM_FOR_PNET=50

'''
RNET 网络区

'''
#RNET数据处理
RNET_DATASET_PATH='/home/zhangxk/AIProject/MTCNN_TRAIN/rnet/dataset'
NMS_DEFAULT=[0.5,0.7]
PNET_THREAD=0.3
NEG_NUM_FOR_RNET=60