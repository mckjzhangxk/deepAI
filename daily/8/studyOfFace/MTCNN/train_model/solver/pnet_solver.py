from utils.dbutils import get_example_nums
from Configure import PNET_DATASET_PATH

model_name='PNet'
IMG_SIZE=12
##############################
EXAMPLES=1000000
# EXAMPLES=get_example_nums(PNET_DATASET_PATH,'PNet.txt')
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
MODEL_LOG_DIR='/home/zhangxk/AIProject/MTCNN_TRAIN/pnet/log'
MODEL_CHECKPOINT_DIR='/home/zhangxk/AIProject/MTCNN_TRAIN/pnet/model/'