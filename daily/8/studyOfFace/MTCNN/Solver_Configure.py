IMG_SIZE=12
##############################
EXAMPLES=1000000
EPOCH=3000
BATCH_SIZE=384
LoopPerEpoch=EXAMPLES//BATCH_SIZE+1
#在第LR_EPOCH 学习率衰减
LR=1E-4
DECAY_FACTOR=0.9
LR_EPOCH = [8000]
# 循环10次显示一次LOSS
DISPLAY_EVERY=10

#日志信息
MODEL_LOG_DIR='/home/zhangxk/AIProject/MTCNN_TRAIN/pnet/log'
MODEL_CHECKPOINT_DIR='/home/zhangxk/AIProject/MTCNN_TRAIN/pnet/model/'