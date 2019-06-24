import json
import collections

if __name__ == '__main__':
    config=collections.OrderedDict({
        'use_tpu':True,
        'tpu_name':None,
        'tpu_zone':None,
        'tpu_iterations':0,
        'tpu_master':None,
        'model_dir':None,
        'train_batch_size':128*8,
        'eval_batch_size':1000,
        'predict_batch_size':1000
    })
    
    with open('config.json',mode='w',encoding='utf-8') as fp:
        json.dump(config,fp,indent=1)