#训练P网络数据的基数
BASE_NUM=1000
#TRAINSET
WIDER_TRAINSET='/home/zhangxk/AIProject/WIDER_train/images'
WIDER_ANNOTION='/home/zhangxk/AIProject/WIDER_train/wider_face_train_bbx_gt_test.txt'


#PNET数据处理参数,
PNET_DATASET_PATH= '/home/zhangxk/AIProject/MTCNN_TRAIN/pnet/dataset'
POSITIVE_COPYS=20
NEGATIVE_COPYS=5
NEG_NUM=50