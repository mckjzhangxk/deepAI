import numpy as  np
from sklearn.cluster import KMeans
def quanizer(X,codebook):
    '''
    
    :param X: (N,d)
    :param coodbook:(z,d/m) 
    :return: (N,m)
    '''

    def _distance(a,b):
        '''
        
        :param a:(N,m,dm) 
        :param b:(m,z,dm)
        :return: (N,m,z)
        '''
        a=np.expand_dims(a,-2)
        r=np.sum((a-b)**2,axis=-1)
        return r

    assert X.ndim==1 or X.ndim==2,"输入必须shape=(d,) or(N,d)"
    assert codebook.ndim==3,"codebook必须shape=(m,z,dm)"

    d=X.shape[-1]
    dreduce=codebook.shape[-1]
    m=codebook.shape[0]
    assert d % dreduce==0 and d//dreduce==m,"X的维度必须是codebook的维度的倍数"


    if X.ndim==0:
        X=np.expand_dims(X,axis=0)
    X=X.reshape((-1,m,dreduce))

    dist=_distance(X,codebook)

    ret=np.argmin(dist,-1)
    return ret


def createCodeBook(X,m=64,z=256,outpath='codebook.npy',cpus=1,max_iter=300):
    import tqdm
    d=X.shape[-1]
    assert d%m==0,"m必须是d 的因数"
    dreduce=d//m
    X=X.reshape((-1,m,dreduce))

    codebook=[]

    for im in tqdm.tqdm(range(m)):

        model=KMeans(z,n_jobs=cpus,verbose=1,max_iter=max_iter)
        model.fit(X[:,im,:])
        codebook.append(model.cluster_centers_)
    codebook=np.array(codebook)

    assert codebook.shape==(m,z,dreduce)
    np.save(outpath,codebook)

if __name__ == '__main__':
    N=300
    d=512
    m=64
    z=256

    # x=np.random.rand(N,d)
    # # c=np.random.rand(m,z,d//m)
    # c=np.load('codebook.npy')
    # print(c.shape)
    # r=quanizer(x,c)
    #
    # assert r.shape==(N,m)

    import bcolz

    x=bcolz.open('/home/zxk/桌面/dump/feature','r')
    x=x[:]
    print(x.shape)
    # x=np.random.rand(N,d)
    createCodeBook(x,m=m,z=z,cpus=-1)