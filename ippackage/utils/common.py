import sys
import time
import numpy as np

def progess_print(info):
    sys.stdout.write('\r>>'+info)
    sys.stdout.flush()

def print_state_info(stat_info):
    endtime=time.time()
    step=stat_info['steps']
    avg_loss=np.average(stat_info['total_loss'])
    avg_acc=np.average(stat_info['accuracy'])
    speed=stat_info['stat_steps']/(endtime-stat_info['start_time'])
    lr=stat_info['learn_rate']

    _s='step %d,average loss is %.3f,accuracy is %.3f,lr:%f,speed %.2f.'%\
       (step,avg_loss,avg_acc,lr,speed)
    print(_s)
