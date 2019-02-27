from data.utils import load_data,connectId2tuple
from data.Beans import Package
from data.analysisUtils import pca,standardizeData,fourierAnalysis,save_fig,show_distribution
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans


def get_dataset(path, features, normData=True, FourTransform=True, windowSize=None, mv_avg=10, beanFunc=Package):
    '''
    path里面记载了一段时间区间里的通信记录
    本方法读出一行记录,使用beanFunc把记录转化成一个描述通信信息的对象P(表示一个时间点,对某一个通信的记录),
    记录到数据库DB中,
    DB的结构:
        db:(N,T,D)的array实际的数据
        getConnectId([id1,id2]):返回idi对应的通信表示
        search(connectid):返回和connectid对于的[id1,id2....],这里采用默认模糊查询
    本方法还会:
        1.对DB.db做标准化处理(normData=True)
        2.对DB.db,使用选用windowSize的窗口大小,分段分析特征的频谱(DFT)
    返回:
        db:DB对象
        np_db:数据数据处理后的db.db
    :param path: 
    :param features: 
    :param normData: 
    :param FourTransform: 
    :param windowSize: 
    :return: 
    np_db:(N,T,D),D=len(features),如果FourTransform=False,表示原始特征.否则是FFT换后的特征,
    对于FourTransform=False,np_db[:,t,d]表示在t秒的d特征的通信信息
    '''
    db = load_data(path, features, beanFunc)
    np_db = db.db
    # v=np.ones((mv_avg))/mv_avg
    # N,T,D=np_db.shape
    # for n in range(N):
    #     for d in range(D):
    #         np_db[n,:,d]=np.convolve(np_db[n,:,d],v,'same')

    if normData:
        np_db = standardizeData(np_db, 'STD')
    if FourTransform:
        np_db, feature_size, steps = fourierAnalysis(np_db, windowSize=windowSize)
        np_db = np.abs(np_db)
    return db, np_db


def findUnusual(data):
    '''
    
    :param data: (N,D)的array
    :return: 使用kmean算法聚类,K=2,返回数目小的一类索引下标,(N',)
    '''
    kmeans = KMeans(init='k-means++', n_clusters=2, n_init=10)
    kmeans.fit(data)
    clusters = kmeans.labels_
    N = len(data)
    e0, e1 = np.sum(clusters == 0), np.sum(clusters == 1)
    # print('Total %d examples,positive %d ,negative %d' % (N, e1,e0))

    unusual_cluster = np.argmin([e0, e1])
    unusual_connection_index = np.argwhere(clusters == unusual_cluster)
    unusual_connection_index = np.squeeze(unusual_connection_index, axis=1)

    b = kmeans.cluster_centers_[unusual_cluster]
    a = data[unusual_connection_index]
    inertia = np.average(np.linalg.norm(a - b, axis=1))
    return unusual_connection_index, inertia


def run_analysis(data, db, window=300, threshold=0.2):
    '''
    datas是要分析的数据(N,T,D),使用window大小的窗口进行时序
    聚类分析,需要分析的时序=step=T//window
    每一次时序分析都会得出一个index的数组,
    表示当前时序异常的索引.
    
    通过db.getConnectId可以反查是哪一个通信,
    把这个通信ID转成(ipA,ipB),保存到一个数组stats中
    
    时序分析结束后,stat保存的是所有时序挑选出的异常通信索引.
    
    最后对上个数组stats统计,得票多的约有可能是异常通信(VPN,FREEGate)
    
    备注A->B和B->A表示一个通信,用(A,B)
    
    返回:list(tuple,score)
        tuple(commucateA,commucateB)
        score:这个通信是异常的得分
    
    :param data:(N,T,D)
    :param window: 
    :return: 
    '''
    N, T, D = data.shape
    stats = []
    steps = 0
    for t in range(0, T, window):
        data_t = data[:, t:t + window, :]
        feed_data = np.reshape(data_t, (N, -1))
        _index, intera = findUnusual(feed_data)
        connectids = db.get_connect_ID(_index)
        for connectid in connectids:
            stats.append(connectId2tuple(connectid))
        steps += 1
    stats = Counter(stats)

    ret=[]
    for connectid, score in stats.most_common(10):
        score = 1.0 * score / steps
        if score > threshold:
            ret.append((connectid, score))
    return ret