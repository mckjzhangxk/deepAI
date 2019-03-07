import os
import numpy as np
class DataConf():
    #source_path下包含所有数据,txt格式
    source_path = '/home/zhangxk/AIProject/ippack/vpndata/out'
    #把source_path每Tmax进行一个归组,输出到target_path/{group_index}下面
    target_path = '/home/zhangxk/AIProject/ippack/vpndata/out_T'
    #临时输出的数据文件
    data_path = '/home/zhangxk/AIProject/ippack/vpndata/data/3-6.txt'

    features = ['upcount', 'upsize', 'up_rate', 'downcount', 'downsize', 'down_rate']
    features_func=[None,
                  lambda x:np.log10(x+1e-14),
                   None,
                   None,
                   lambda x: np.log10(x + 1e-14),
                   None]
    Tmax = 300

    #黑名单,用于标记数据
    BLACK_LIST_PATH = '/home/zhangxk/projects/deepAI/ippackage/data/resource/blacklist'
    with open(BLACK_LIST_PATH) as fs:
        lines = fs.readlines()
        BLACK_LIST = [l.strip('\n') for l in lines]
        BLACK_LIST=set(BLACK_LIST)

    #######################训练数据与测试数据的准备###############################

    BASE_NUM=10000
    TrainDataSource='/home/zhangxk/AIProject/ippack/vpndata/data'
    TrainFile='/home/zhangxk/AIProject/ippack/vpndata/train.txt'
    DevDataSource = '/home/zhangxk/AIProject/ippack/vpndata/data'
    DevFile = '/home/zhangxk/AIProject/ippack/vpndata/dev.txt'
    Dataset_Log='/home/zhangxk/AIProject/ippack/vpndata/readme.txt'
