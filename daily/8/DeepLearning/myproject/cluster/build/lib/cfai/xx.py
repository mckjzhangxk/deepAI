import traceback
import uuid
import os
import json

def handleErrorLog(logpath,errobj):
    '''
    
    :param logpath: 输出的log路径
    :param errobj: 要输出的errobj对象
    :return: 返回错误编号
    '''
    uid=str(uuid.uuid1()).replace('-', '')
    outpath=os.path.join(logpath,uid+'.json')
    errinfo=traceback.format_exc()


    with open(outpath, 'w') as fp:
        json.dump(errobj, fp, indent=1)
        fp.write(errinfo)

    return uid
