from data.movedata import move
from data.dbutils import get_package_info
from data.Beans import Package_FreeGate
import collections
import os
import numpy as np
from utils.common import progess_print

source_path='/home/zhangxk/AIProject/ippack/ip_capture/out'
target_path='/home/zhangxk/AIProject/ippack/ip_capture/hello'
train_path=os.path.join(target_path,'train1.txt')

features=['upcount','upsize','up_rate','downcount','downsize','down_rate']
features=['upcount']
Tmax=15

class VPN_Record(collections.namedtuple('VPN_Record',['label','length','features'])):pass

def _get_valid_input(series):
    '''
    series是一个list,表示时间序列,但是这个序列头尾
    可能会被大量的None填充,这些视为无效输入,因为可能
    这段实际根本就不存在链接通信.

    返回:l=serise[a,b],l[0]和l[-1]不为None,
    l的长度是有效序列的长度
    :param series: 
    :return: 
    '''
    s = 0
    for k in series:
        if k is None:
            s += 1
        else:
            break
    e = len(series)
    for k in reversed(series):
        if k is None:
            e -= 1
        else:
            break
    assert e > s, 'at least one element is not None'
    return series[s:e]


def __extract__features(package_T,features,Tmax):
    '''
    提取一条记录的特征,特征名由features[]给出,特征值保存在package_T中,
    package_T是长度为T1的list,package_T[t]是t时刻的特征,把package_T填充
    成Tmax的序列
    
    返回:array(Tmax,D)
    D=len(features)
    
    :param package_T: 
    :param features: 
    :param Tmax: 
    :return: 
    '''

    Ts = []

    for pack_t in package_T:
        feature_values = []
        Ts.append(feature_values)
        for feature_name in features:
            if pack_t is None:
                feature_values.append(0.0)
            else:
                assert hasattr(pack_t, feature_name), 'object do not have feature %s' % feature_name
                feature_values.append(getattr(pack_t, feature_name))
    for t in range(Tmax-len(package_T)):
        feature_values = []
        Ts.append(feature_values)
        for feature_name in features:
            feature_values.append(0.0)
    Ts=np.array(Ts)

    return Ts

def dict_to_record(d,features=[],Tmax=0):
    '''
    d是dict,key表示一个链接,值是有效的时间序列list,
    这个把d转化成一个文件的记录,记录的格式是VPN_Record类型
    
    :param d: 
    :param features: 
    :param Tmax: 
    :return: 
    '''
    ret={}
    N=len(d)
    for s,(cid,pack) in enumerate(d.items()):
        _feature=__extract__features(pack,features,Tmax)
        _label='0'
        _length=len(pack)
        _record=VPN_Record(_label,_length,_feature)
        ret[cid]=_record
        if s% 5==0:
            progess_print('dict_to_record %d/%d'%(s,N))
    print()
    return ret

def write_record(d,output_file):
    with open(output_file,'a+') as fs:
        for connectid,record in d.items():
            _tmp=[record.label,record.length]
            #这里还可以对数据做缩放处理
            _f=list(np.reshape(record.features,(-1)))
            _tmp=_tmp+_f
            _tmp=map(str,_tmp)
            _r=','.join(_tmp)+'\n'
            fs.write(_r)

if __name__ == '__main__':
    #第一步:输入数据分割成splits份,输出到target_path中,
    # 这样不至于数据量过大不好处理
    outs=move(source_path, target_path, Tmax)
    #第二部,在outs[i]的文件是时间[Tmax*i,Tmax*(i+1)]
    # 内的流量统计信息,把大量文件转化成dict,然后取有效序列
    for p in outs:
        #p是一个文件夹,把p文件夹下面的文件转成dict,key是connectid,value[package],
        #这里的value是定长Tmax的数组
        _t=get_package_info(p,Package_FreeGate)
        #提取出有效的values,不同key是connectid对于不同长度的values
        for k,v in _t.items():
            _t[k]=_get_valid_input(v)
        #再把上面的dick转化成一个固定格式的record,0会填充Trel<Tmax的部分
        vv=dict_to_record(_t,features,Tmax)
        write_record(vv,train_path)