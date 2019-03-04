import  os,shutil
from utils.common import progess_print
import numpy as np

def  move(source_path, target_base_path, T):
    '''
    source_path里面保存了连续时刻的文件,
    每个T个文件,进行一次分割,输出到target_path/{t}
    目录下面
    :param source_path: 
    :param target_base_path: 
    :param T: 
    :return: 
    '''

    print('start split data')
    filelist=os.listdir(source_path)
    filelist=sorted(filelist)

    numOfFile=len(filelist)
    numGrp=int(np.ceil(numOfFile/T))

    ret=[]
    for gid in range(numGrp):

        target_path=os.path.join(target_base_path, str(gid))
        if not os.path.exists(target_path):
            os.mkdir(target_path)
        ret.append(target_path)
        for k in range(gid*T,min(gid*T+T,numOfFile)):
            srcfile=os.path.join(source_path,filelist[k])
            tgtfile=os.path.join(target_path,filelist[k])


            shutil.copy(srcfile,tgtfile)
        if gid %2==0:
            progess_print('finish split data %d/%d'%(gid,numGrp))
    print()
    return ret