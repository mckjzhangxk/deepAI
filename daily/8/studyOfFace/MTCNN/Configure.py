'''

公共区域,数据集路径
'''
#TRAINSET
WIDER_TRAINSET='/home/zhangxk/AIProject/WIDER_train/images'
WIDER_ANNOTION='/home/zhangxk/AIProject/WIDER_train/wider_face_train_bbx_gt_test.txt'
#LWF用于landmark
LWF_TRAINSET='/home/zhangxk/AIProject/lfw'
LWF_ANNOTION='/home/zhangxk/AIProject/lfw/trainImageList_test.txt'


FACE_MIN_SIZE=50 #最小人脸尺寸
SCALE=0.79    #人脸金字塔,每次缩放比例
# DETECT_EPOCHS=[3,4,5]  #生成hardexample时候,P,R,O分别使用第EPOCH?个训练的模型
DETECT_EPOCHS=[440,2,2]
#PNET使用2次nms,RNET,ONET各使用一次
NMS_DEFAULT=[0.5,0.7,0.6,0.6]
THRESHOLD=[0.3, 0.1, 0.4]
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
#一张图片生成最多NEG_NUM_FOR_RNET张人脸
NEG_NUM_FOR_RNET=60
'''

'''
#RNET数据处理
ONET_DATASET_PATH='/home/zhangxk/AIProject/MTCNN_TRAIN/onet/dataset'
#一张图片生成最多NEG_NUM_FOR_RNET张人脸
NEG_NUM_FOR_ONET=60