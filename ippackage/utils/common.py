import sys
import time

def progess_print(info):
    sys.stdout.write('\r>>'+info)
    sys.stdout.flush()

def print_state_info(stat_info):
    endtime=time.time()
    step=stat_info['steps']
    avg_loss=stat_info['total_loss']/stat_info('stat_steps')
    avg_acc=stat_info['accuracy']/stat_info('stat_steps')
    speed=(endtime-stat_info['start_time'])/stat_info('stat_steps')
    lr=stat_info('learn_rate')

    _s='step %d,average loss is %.3f,accuracy is %.3f,lr:%f,speed .2f%'%\
       (step,avg_loss,avg_acc,lr,speed)
    print(_s)
