from utils.dbutils import get_example_nums
from Configure import RNET_DATASET_PATH,RNET_DATASET_VALID_PATH,BASE_DIR
from os.path import join


model_name='RNet'
IMG_SIZE=24
##############################
# EXAMPLES=100000
EXAMPLES=get_example_nums(RNET_DATASET_PATH,'pos.txt')*6
VALID_EXAMPLES=get_example_nums(RNET_DATASET_VALID_PATH,'RNet.txt')

EPOCH=100
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
CHECK_GRADIENT=False
#日志信息
MODEL_LOG_DIR=join(BASE_DIR,'MTCNN_TRAIN/rnet/log')
MODEL_CHECKPOINT_DIR=join(BASE_DIR,'MTCNN_TRAIN/rnet/model')
MODEL_RECOLVER_PATH=join(BASE_DIR,'MTCNN_TRAIN/rnet/model-19-1-31/RNet-29')
