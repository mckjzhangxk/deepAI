from utils.dbutils import get_example_nums
from Configure import RNET_DATASET_PATH
model_name='RNet'
IMG_SIZE=24
##############################
EXAMPLES=1000000
# EXAMPLES=get_example_nums(RNET_DATASET_PATH)

EPOCH=3000
BATCH_SIZE=384
LoopPerEpoch=EXAMPLES//BATCH_SIZE+1
#在第LR_EPOCH 学习率衰减
LR=1E-4
DECAY_FACTOR=0.9
#LR_EPOCH表示在那几个EPOCH decay
LR_EPOCH = [8000]
# 循环10次显示一次LOSS
DISPLAY_EVERY=10

#日志信息
MODEL_LOG_DIR='/home/zhangxk/AIProject/MTCNN_TRAIN/rnet/log'
MODEL_CHECKPOINT_DIR='/home/zhangxk/AIProject/MTCNN_TRAIN/rnet/model/'