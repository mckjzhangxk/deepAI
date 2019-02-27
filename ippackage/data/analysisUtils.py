import numpy as np
import matplotlib.pyplot as plt


def show_distribution(arr,featureId,featureName,basedir=None):
    '''
    
    :param arr:(T,D)的数组 
    :param featureId: [0,D)
    :return: 
    '''
    plt.figure()
    arr=arr[:, featureId]
    plt.plot(range(len(arr)),arr)
    plt.xlabel('time')
    plt.ylabel('value')
    plt.savefig('%s/%s_time.jpg' % (basedir, featureName))
    arr=arr[arr>0]

    plt.figure()
    plt.hist(arr)
    plt.xlabel('values')
    plt.ylabel('distribution')
    _u=np.mean(arr)
    _std=np.std(arr)
    plt.title('%s,mean:%.2f,std:%.2f'%(featureName,_u,_std))
    plt.savefig('%s/%s_dist.jpg'%(basedir,featureName))

def save_fig(mat,path,logScale=False):
    plt.figure()
    if logScale:
        mat=np.log10(mat+1e-14)
    plt.matshow(mat)
    plt.colorbar()
    plt.savefig(path)

def pca(a):
    shape_a=a.shape
    a=a.reshape(-1,shape_a[-1])

    U, S, VT = np.linalg.svd(a, False)
    pri_axis=VT[0]*np.sign(VT[0][0])
    print(pri_axis,np.linalg.norm(pri_axis))
    a=a.dot(pri_axis)
    a=a.reshape(*(shape_a[:-1]))
    return a
def standardizeData(d,mode=None):
    def _standardize1(d):
        m = np.max(d, axis=0, keepdims=True) + 1e-14
        return d / m

    def _standardize2(d):
        u = np.mean(d, axis=0, keepdims=True)*0
        s = np.std(d, axis=0, keepdims=True) + 1e-14
        return (d - u) / s

    if mode=='STD':
        return _standardize2(d)
    if mode=='MAX':
        return _standardize1(d)
    return d


def _stft(x,windowSize):
    '''
    
    :param x: shape(N,T)
    :param windowSize:窗口大小
    :return: shape(N,T//windowSize*windowSize),时序分析结果
    
    '''
    steps=(x.shape[1]//windowSize)
    x_len=steps*windowSize
    feature_size=(windowSize//2)+1
    x=x[:,:x_len]

    ret=[]
    for s in range(steps):
        _x=x[:,s*windowSize:s*windowSize+windowSize] #shape(N,windowSize)
        ret.append(_fft(_x)[:,0:feature_size])
    ret=np.concatenate(ret,axis=1)
    return ret

'''
x:shape(N,t),沿着t轴做变化,得到频谱
'''
def _fft(x):
    return np.fft.fft(x,axis=-1)


def fourierAnalysis(X,windowSize):
    '''
    对X 进行stft分析,返回的Y:shape(N,T',D)
    D表示特征数量
    N是批处理数量
    T':steps=(T)//windows,S=(windows//2)+1,
        T'=S*s,对一段序列分析的特征长度是S,序列长度是steps
    
    Y[i,t,j]表示第i个example的j特征
    s=t//S表示是对s段序列的分析,对于区间[s*windowSize,s*windowSize+windowSize]
    k=t%S,第K个频率的分析
    (T1,T2.......Tsteps),Ti have shape windowSize
                         _______________
                        |               \
                        |               \____
     (windowSize,)---> |magic machine    \   >(featureSize,)
                      |__________________\
        
    :param X:(X,T,D) 
    :param windowSize: 
    :return: Y(N,T',D),featureSize一个序列特征的大小
    
    '''
    N,T,D=X.shape
    if windowSize==None:windowSize=1
    steps=T//windowSize
    featureSize=(windowSize//2)+1
    Y=[]
    for d in range(D):
        sf=_stft(X[:,:,d],windowSize)
        Y.append(sf)
    Y=np.stack(Y,axis=-1)

    return Y,featureSize,steps