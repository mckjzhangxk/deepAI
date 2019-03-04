import os

import numpy as np

from data.Beans import Package
from utils.common import progess_print

'''
从单个文件里面读取一秒内的流量情况,输出一个dict,
每一条记录表示一次通信记录,保存了:
    key:通信的标识srcip:srcport->tgtip:tgtport_type
    value:Package对象关于通信的信息
    src(ip,port)
    tgt(ip,port)
    protcal_type
    timestamp
    upload_count,update_data_size-->upload_rate
    download_count,downdate_data_size-->download_rate
'''

def connectId2tuple(connectid):
    sps=connectid.split('->')
    src=sps[0]
    sps=sps[1].split('_')
    dest=sps[0]
    ptype=sps[1]

    if src<dest:
        return (src,dest,ptype)
    else:
        return (dest,src,ptype)

def _read_single_file(filename,bean):
    '''
    filaname是一秒钟libpcap抓取的数据包日志文件,
    把这个文件转化成一个dict,key标识一个链接,规则是
    根据bean.signature函数决定,value是bean对象,
    记录了在这一秒内,有这个链接上传速度,下载速度,上传次数
    下载次数,时间戳信息...
    
    返回dict..d
    
    :param filename: 
    :param bean: 
    :return: 
    '''
    '''
    arr:a list of Package object
    arr里面的数据都没有统计下载信息,这里会更新下载信息
    '''

    def __update_downloadinfo__(arr):
        tmp_signature = {}
        for p in arr:
            sig = p.signature_connection
            if sig in tmp_signature:
                tmp_signature[sig].append(p)
            else:
                tmp_signature[sig] = [p]
        for pair in tmp_signature.values():
            assert len(pair) <= 2, 'error,connect pair at most length 2'
            if len(pair) == 1: continue
            a = pair[0]
            b = pair[1]
            a.set_downloadinfo(b)
            b.set_downloadinfo(a)

    fs=open(filename,'r')
    lines=fs.readlines()
    ret={}
    records=map(lambda s:s.split(),lines)
    for r in records:
        _size=int(r[6])
        _ts=r[0]
        #字段含义:时间戳,源IP,目标IP(2),源端口(3),目标端口(4),协议号(5),数据包大小(6)
        p = bean(
            srcip=r[1],
            srcport=r[3],
            destip=r[2],
            destport=r[4],
            type=r[5],
            upcount=1,
            upsize=_size,
            ts=_ts)
        _signature=p.signature
        if _signature in ret:
            _p=ret[_signature]
            _p.upcount+=1
            _p.upsize+=_size
        else:
            ret[_signature]=p
    __update_downloadinfo__(ret.values())
    return ret


def get_package_info(basepath, bean):
    '''
    basepath如果有T个文件,说明在[start_time,start_time+T)时间段内的
    通信信息全部记录在本文件夹下面,对通信按照时序进行统计
    key:srcip:srcport->destip:destport:type 是一个通信链路的ID
    value:list(T){Package}:长度是T的list,每个元素是Package,value[t]是对
    start_timet+t秒的流量统计
    
    :param basepath: 区间为T的通信记录,每一秒保存到一个文件里面
    :return: dict{connectId,list(T)[Package]}
    '''
    def __updatedict__(base_dict,new_dict,t,T):
        '''
        在t时刻的一条链接信息new_dict,new_dict.key是链接标识
        new_dict.value是具体信息
        
        :param base_dict: 
            key:src->dest的签名
            values:list(T)
        :param new_dict: new_dict时间t的通信记录
        :param t: 当前处理t
        :param T: 总长T
        :return: 
        '''
        for k,v in new_dict.items():
            info=base_dict.get(k,[None]*T)
            info[t]=v
            base_dict[k]=info

    flist=os.listdir(basepath)
    flist=sorted(flist)
    T=len(flist)

    ret={}

    for t,fname in enumerate(flist):
        filepath=os.path.join(basepath,fname)
        info_t=_read_single_file(filepath,bean)
        __updatedict__(ret,info_t,t,T)
        if t%10==5:
            progess_print('file to beans %d/%d'%(t,T))
    print()

    return ret

def _extract_features(info, feature_names=[]):
    '''
    
    :param info:get_package_info返回的对区间流量的统计信息 
    :param feature_names: list of feature_name
    :return: dict
        key:connectId
        value:list of selected feature,每个元素是(T,D)的array
    '''
    ret={}
    T=len(info)
    #package_T is list of Package Object
    for t,(connectID,package_T) in enumerate(info.items()):
        Ts=[]

        for pack_t in package_T:
            feature_values=[]
            Ts.append(feature_values)
            for feature_name in feature_names:
                if pack_t is None:feature_values.append(0.0)
                else:
                    assert hasattr(pack_t,feature_name),'object do not have feature %s'%feature_name
                    feature_values.append(getattr(pack_t,feature_name))
        ret[connectID] = np.array(Ts)
        if t%10==5:
            progess_print('finish extract_features %d/%d'%(t,T))
    print()
    return ret

class DB():
    def __init__(self,data,feature_names=None):
        '''
            把data{connectId,array(T,D)} 转化成
            _db_index:{connectId:connectidx}
            _db:array(N,T,D)
            
            例如想找到(192.168.0.12->8.8.8.8)这条通信的信息,
            idex=_db_index['192.168.0.12->8.8.8.8']
            info=_db[idex]
        :param data: 
        :param feature_names: 
        '''
        keys=sorted(data.keys())

        self._db = np.array([data[k] for k in keys])
        self._db_index = {v: i for i, v in enumerate(keys)}
        self._db_inv_index={v:k for k, v in self._db_index.items()}
        self._feature_names = feature_names
    def __len__(self):
        return len(self._db_index)
    @property
    def db(self):return self._db
    @property
    def db_index(self):return self._db_index
    @property
    def db_inv_index(self):return self._db_inv_index
    def _getConnectId(self,connectId):
        return self._db_index.get(connectId, -1)
    def search(self,connectId,precise=False):
        '''
        
        :param connectId: 
        :param precise: True精准查询,False模糊查询
        :return: 精准查询会精准匹配connectId,返回
            None:没有查到
            array:(T,Dfeatures)
        模糊查询的返回:
            indexes:[]
            _db_index:[array:(T,Dfeatures),array:(T,Dfeatures),array:(T,Dfeatures)]
        '''
        if precise:
            index=self._getConnectId(connectId)
            return index if index >= 0 else None
        else:
            indexs=self._search(connectId)
            return indexs

    def _search(self,connectinfo):
        ret=[]

        for connectid,index in self._db_index.items():
            if connectinfo in connectid:
                ret.append(index)
        ret=sorted(ret)
        return ret

    def get_connect_ID(self,indexes):
        if isinstance(indexes,np.ndarray):
            indexes=list(indexes)
        if not isinstance(indexes,list):
            indexes=[indexes]
        return [self._db_inv_index[i] for i in indexes]


def load_data(path,feature_names=[],bean=Package):
    pack_infos=get_package_info(path, bean)
    pack_infos=_extract_features(pack_infos,feature_names=feature_names)
    return DB(pack_infos,feature_names)

def printutils(kk,filter_func=None):
    for k,v in kk.items():
        if filter_func is None or filter_func(k):
            print(k)
            print(v)
