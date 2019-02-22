from data.utils import load_data,connectId2tuple
from data.analysisUtils import pca,standardizeData,fourierAnalysis,save_fig,show_distribution
import numpy as np
import os
from collections import Counter
from sklearn.cluster import KMeans



def get_dataset(path,features,normData=True,FourTransform=True,windowSize=None,mv_avg=10):
    '''
    path里面记载了一段时间区间里的通信记录,本方法把一个srcip:srcport->descip:destport_type
    作为一个签名,并对通信时间排序后得到np_db,并把原始信息记录在db里面
    :param path: 
    :param features: 
    :param normData: 
    :param FourTransform: 
    :param windowSize: 
    :return: 
    np_db:(N,T,D),D=len(features),如果FourTransform=False,表示原始特征.否则是FFT换后的特征,
    对于FourTransform=False,np_db[:,t,d]表示在t秒的d特征的通信信息
    '''
    db = load_data(path, features)
    np_db = db.db
    # v=np.ones((mv_avg))/mv_avg
    # N,T,D=np_db.shape
    # for n in range(N):
    #     for d in range(D):
    #         np_db[n,:,d]=np.convolve(np_db[n,:,d],v,'same')

    if normData:
        np_db=standardizeData(np_db,'STD')
    if FourTransform:
        np_db, feature_size, steps = fourierAnalysis(np_db, windowSize=windowSize)
        np_db = np.abs(np_db)
    return db,np_db

def findUnusual(data):
    '''
    
    :param data: (N,D)的array
    :return: 使用kmean算法聚类,K=2,返回数目小的一类索引下标,(N',)
    '''
    kmeans = KMeans(init='k-means++', n_clusters=2, n_init=10)
    kmeans.fit(data)
    clusters = kmeans.labels_
    N=len(data)
    e0,e1=np.sum(clusters == 0),np.sum(clusters == 1)
    # print('Total %d examples,positive %d ,negative %d' % (N, e1,e0))

    unusual_cluster=np.argmin([e0,e1])
    unusual_connection_index = np.argwhere(clusters == unusual_cluster)
    unusual_connection_index = np.squeeze(unusual_connection_index, axis=1)

    b=kmeans.cluster_centers_[unusual_cluster]
    a=data[unusual_connection_index]
    inertia=np.average(np.linalg.norm(a-b,axis=1))
    return unusual_connection_index,inertia


def slice(data,db,perodic=300):
    '''
    
    :param data:(N,T,D)
    :param perodic: 
    :return: 
    '''
    N,T,D=data.shape
    stats=[]
    steps=0
    for t in range(0,T,perodic):
        data_t=data[:,t:t+perodic,:]
        feed_data=np.reshape(data_t,(N,-1))

        _index, intera = findUnusual(feed_data)
        connectids=db.get_connect_ID(_index)
        for connectid in connectids:
            stats.append(connectId2tuple(connectid))
        steps+=1
    stats=Counter(stats)
    print('steps:',steps)
    print(stats.most_common(10))
'''
out-2-18_1-->(normalize,fft)=(T,T),(T,F)
['010.010.010.223:137->010.255.255.255:137_17', '010.010.030.119:137->010.010.030.255:137_17', '172.016.005.255:137->172.016.255.255:137_17', '192.168.020.172:137->192.168.020.255:137_17', '192.168.060.058:137->192.168.060.255:137_17', '192.168.060.150:137->192.168.060.255:137_17', '192.168.060.175:137->192.168.060.255:137_17', '192.168.060.182:137->192.168.060.255:137_17']
['010.010.010.223:137->010.255.255.255:137_17', '010.010.030.119:137->010.010.030.255:137_17', '172.016.005.255:137->172.016.255.255:137_17', '192.168.020.172:137->192.168.020.255:137_17', '192.168.060.058:137->192.168.060.255:137_17', '192.168.060.150:137->192.168.060.255:137_17', '192.168.060.175:137->192.168.060.255:137_17', '192.168.060.182:137->192.168.060.255:137_17']
['192.168.060.158:35848->150.138.250.048:443_6']
['192.168.060.158:35848->150.138.250.048:443_6']
out-2-18_2-->(normalize,fft)=(T,T),(T,F)
['045.041.188.071:49181->192.168.060.158:41473_17', '192.168.060.158:41473->045.041.188.071:49181_17']
['045.041.188.071:49181->192.168.060.158:41473_17', '192.168.060.158:41473->045.041.188.071:49181_17']
['192.168.060.158:41473->045.041.188.071:49181_17']
['192.168.060.158:41473->045.041.188.071:49181_17']

out-2-15_1-->(normalize,fft)=(T,T),(T,F),windowSize=300
['000.000.000.000:68->255.255.255.255:67_17', '010.010.010.223:137->010.255.255.255:137_17', '010.010.030.119:137->010.010.030.255:137_17', '045.041.185.242:19310->192.168.060.158:47166_17', '172.016.005.255:137->172.016.255.255:137_17', '192.168.003.011:5353->224.000.000.251:5353_17', '192.168.003.012:5353->224.000.000.251:5353_17', '192.168.003.013:5353->224.000.000.251:5353_17', '192.168.003.014:5353->224.000.000.251:5353_17', '192.168.020.172:137->192.168.020.255:137_17', '192.168.020.186:1609->255.255.255.255:20001_17', '192.168.060.040:137->192.168.060.255:137_17', '192.168.060.150:137->192.168.060.255:137_17', '192.168.060.158:47166->045.041.185.242:19310_17', '192.168.060.182:137->192.168.060.255:137_17', '192.168.060.190:137->192.168.060.255:137_17']
['010.010.010.223:137->010.255.255.255:137_17', '010.010.030.119:137->010.010.030.255:137_17', '045.041.185.242:19310->192.168.060.158:47166_17', '172.016.005.255:137->172.016.255.255:137_17', '192.168.020.172:137->192.168.020.255:137_17', '192.168.060.150:137->192.168.060.255:137_17', '192.168.060.158:47166->045.041.185.242:19310_17', '192.168.060.182:137->192.168.060.255:137_17', '192.168.060.190:137->192.168.060.255:137_17']
['192.168.060.158:47166->045.041.185.242:19310_17']
['192.168.060.158:47166->045.041.185.242:19310_17']


out-2-15_2-->(normalize,fft)=(T,T),(T,F),windowSize=300
['000.000.000.000:68->255.255.255.255:67_17', '010.010.010.125:137->010.255.255.255:137_17', '010.010.010.223:137->010.255.255.255:137_17', '010.010.030.119:137->010.010.030.255:137_17', '045.041.185.242:19310->192.168.060.158:47166_17', '172.016.005.255:137->172.016.255.255:137_17', '192.168.003.011:5353->224.000.000.251:5353_17', '192.168.003.012:5353->224.000.000.251:5353_17', '192.168.003.013:5353->224.000.000.251:5353_17', '192.168.003.014:5353->224.000.000.251:5353_17', '192.168.020.128:57466->239.255.255.250:3702_17', '192.168.020.172:137->192.168.020.255:137_17', '192.168.060.040:137->192.168.060.255:137_17', '192.168.060.040:57263->239.255.255.250:1900_17', '192.168.060.075:62805->239.255.255.250:3702_17', '192.168.060.099:49853->239.255.255.250:3702_17', '192.168.060.150:137->192.168.060.255:137_17', '192.168.060.158:47166->045.041.185.242:19310_17', '192.168.060.182:137->192.168.060.255:137_17', '192.168.060.190:137->192.168.060.255:137_17', '192.168.237.173:49293->239.255.255.250:3702_17']
['010.010.010.125:137->010.255.255.255:137_17', '010.010.010.223:137->010.255.255.255:137_17', '010.010.030.119:137->010.010.030.255:137_17', '172.016.005.255:137->172.016.255.255:137_17', '192.168.020.172:137->192.168.020.255:137_17', '192.168.060.150:137->192.168.060.255:137_17', '192.168.060.182:137->192.168.060.255:137_17', '192.168.060.190:137->192.168.060.255:137_17']
['045.041.185.242:19310->192.168.060.158:47166_17', '192.168.060.158:47166->045.041.185.242:19310_17']
['045.041.185.242:19310->192.168.060.158:47166_17', '192.168.060.158:47166->045.041.185.242:19310_17']

out-2-15_3-->(normalize,fft)=(T,T),(T,F),windowSize=300
['000.000.000.000:68->255.255.255.255:67_17', '010.010.010.125:137->010.255.255.255:137_17', '010.010.010.223:137->010.255.255.255:137_17', '010.010.030.119:137->010.010.030.255:137_17', '045.041.185.242:19310->192.168.060.158:47166_17', '172.016.005.255:137->172.016.255.255:137_17', '192.168.003.011:5353->224.000.000.251:5353_17', '192.168.003.012:5353->224.000.000.251:5353_17', '192.168.003.013:5353->224.000.000.251:5353_17', '192.168.003.014:5353->224.000.000.251:5353_17', '192.168.020.172:137->192.168.020.255:137_17', '192.168.060.040:137->192.168.060.255:137_17', '192.168.060.150:137->192.168.060.255:137_17', '192.168.060.158:47166->045.041.185.242:19310_17', '192.168.060.182:137->192.168.060.255:137_17', '192.168.060.190:137->192.168.060.255:137_17']
['010.010.010.125:137->010.255.255.255:137_17', '010.010.010.223:137->010.255.255.255:137_17', '010.010.030.119:137->010.010.030.255:137_17', '172.016.005.255:137->172.016.255.255:137_17', '192.168.020.172:137->192.168.020.255:137_17', '192.168.060.150:137->192.168.060.255:137_17', '192.168.060.182:137->192.168.060.255:137_17', '192.168.060.190:137->192.168.060.255:137_17']
['192.168.060.158:34804->045.041.185.242:19310_17']
['192.168.060.158:34804->045.041.185.242:19310_17']

out-2-19-->(normalize,fft)=(T,T),(T,F),windowSize=300
['010.010.010.125:137->010.255.255.255:137_17', '010.010.010.223:137->010.255.255.255:137_17', '169.254.056.012:137->169.254.255.255:137_17', '172.016.005.255:137->172.016.255.255:137_17', '192.168.020.172:137->192.168.020.255:137_17', '192.168.060.019:137->192.168.060.255:137_17', '192.168.060.150:137->192.168.060.255:137_17', '192.168.060.179:60175->255.255.255.255:137_17', '192.168.060.182:137->192.168.060.255:137_17']
['010.010.010.125:137->010.255.255.255:137_17', '010.010.010.223:137->010.255.255.255:137_17', '169.254.056.012:137->169.254.255.255:137_17', '172.016.005.255:137->172.016.255.255:137_17', '192.168.020.172:137->192.168.020.255:137_17', '192.168.060.019:137->192.168.060.255:137_17', '192.168.060.150:137->192.168.060.255:137_17', '192.168.060.179:60175->255.255.255.255:137_17', '192.168.060.182:137->192.168.060.255:137_17']
['010.010.010.125:137->010.255.255.255:137_17', '010.010.010.223:137->010.255.255.255:137_17', '169.254.056.012:137->169.254.255.255:137_17', '172.016.005.255:137->172.016.255.255:137_17', '192.168.003.011:5353->224.000.000.251:5353_17', '192.168.003.012:5353->224.000.000.251:5353_17', '192.168.003.013:5353->224.000.000.251:5353_17', '192.168.003.014:5353->224.000.000.251:5353_17', '192.168.020.045:62758->192.168.020.142:61477_17', '192.168.020.045:65511->192.168.020.142:52472_17', '192.168.020.128:55676->239.255.255.250:3702_17', '192.168.020.172:137->192.168.020.255:137_17', '192.168.060.019:137->192.168.060.255:137_17', '192.168.060.040:137->192.168.060.255:137_17', '192.168.060.075:62111->239.255.255.250:3702_17', '192.168.060.093:1900->239.255.255.250:1900_17', '192.168.060.093:56039->239.255.255.250:3702_17', '192.168.060.093:62487->239.255.255.250:3702_17', '192.168.060.099:61710->239.255.255.250:3702_17', '192.168.060.100:52861->239.255.255.250:3702_17', '192.168.060.150:137->192.168.060.255:137_17', '192.168.060.179:60175->255.255.255.255:137_17', '192.168.060.182:137->192.168.060.255:137_17', '192.168.060.224:56377->239.255.255.250:3702_17', '192.168.237.173:54732->239.255.255.250:3702_17']
['010.010.010.125:137->010.255.255.255:137_17', '010.010.010.223:137->010.255.255.255:137_17', '169.254.056.012:137->169.254.255.255:137_17', '172.016.005.255:137->172.016.255.255:137_17', '192.168.003.013:5353->224.000.000.251:5353_17', '192.168.003.014:5353->224.000.000.251:5353_17', '192.168.020.128:55676->239.255.255.250:3702_17', '192.168.020.172:137->192.168.020.255:137_17', '192.168.060.019:137->192.168.060.255:137_17', '192.168.060.075:62111->239.255.255.250:3702_17', '192.168.060.093:56039->239.255.255.250:3702_17', '192.168.060.093:62487->239.255.255.250:3702_17', '192.168.060.099:61710->239.255.255.250:3702_17', '192.168.060.100:52861->239.255.255.250:3702_17', '192.168.060.150:137->192.168.060.255:137_17', '192.168.060.179:60175->255.255.255.255:137_17', '192.168.060.182:137->192.168.060.255:137_17', '192.168.237.173:54732->239.255.255.250:3702_17']
'''
path='/home/zhangxk/AIProject/ippack/ip_capture/out'
features=['upcount','upsize','up_rate','downcount','downsize','down_rate']
basepath='/home/zhangxk/projects/deepAI/ippackage/data/debug/1'
db,np_db=get_dataset(path,features,False,False,windowSize=1)
slice(np_db,db,perodic=60)
