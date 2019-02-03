from utils.dbutils import get_example_nums
from Configure import ONET_DATASET_PATH,ONET_DATASET_VALID_PATH,BASE_DIR
from os.path import join

model_name='ONet'
IMG_SIZE=48
##############################
# EXAMPLES=10000
EXAMPLES=get_example_nums(ONET_DATASET_PATH)
VALID_EXAMPLES=get_example_nums(ONET_DATASET_VALID_PATH,'ONet.txt')

EPOCH=30
BATCH_SIZE=384
LoopPerEpoch=EXAMPLES//BATCH_SIZE+1
LoopsForValid=VALID_EXAMPLES//BATCH_SIZE+1
#在第LR_EPOCH 学习率衰减
LR=1E-3
DECAY_FACTOR=0.1
#LR_EPOCH表示在那几个EPOCH decay
LR_EPOCH = [6,12,20]
# 循环10次显示一次LOSS
DISPLAY_EVERY=30
CHECK_GRADIENT=True
#日志信息
MODEL_LOG_DIR=join(BASE_DIR,'MTCNN_TRAIN/onet/log')
MODEL_CHECKPOINT_DIR=join(BASE_DIR,'MTCNN_TRAIN/onet/model/')
MODEL_RECOLVER_PATH=None