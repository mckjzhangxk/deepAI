from utils.dbutils import get_example_nums
from Configure import PNET_DATASET_PATH,PNET_DATASET_VALID_PATH,BASE_DIR
from os.path import join

model_name='PNet'
IMG_SIZE=12
##############################
# EXAMPLES=10000
EXAMPLES=get_example_nums(PNET_DATASET_PATH,'PNet.txt')
VALID_EXAMPLES=get_example_nums(PNET_DATASET_VALID_PATH,'PNet.txt')

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
DISPLAY_EVERY=10

#日志信息
MODEL_LOG_DIR=join(BASE_DIR,'MTCNN_TRAIN/pnet/log')
MODEL_CHECKPOINT_DIR=join(BASE_DIR,'MTCNN_TRAIN/pnet/model/')