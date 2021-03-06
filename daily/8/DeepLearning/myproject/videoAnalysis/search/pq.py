import numpy as  np
from sklearn.cluster import KMeans
from threading import Thread


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

class MyWorker(Thread):
    def __init__(self,id,data,codebook,maxIters=300):
        super().__init__(name=str(id))
        self.model=KMeans(z,n_jobs=1,n_init=1,verbose=1,max_iter=maxIters)
        self.d=data
        self.id=id
        self.codebook=codebook

    def run(self):
        self.model.fit(self.d)
        self.codebook[self.id]=self.model.cluster_centers_
def createCodeBook(X,m=64,z=256,outpath='codebook.npy',max_iter=300,start=0):
    '''
    这个方法 其实是为LSH中对每一段输入做hash准备的hash参数
    
    例如 我的输入是1024维度的float，我相对这个长数组建立索引,建立m=64个索引
    
    也就是每1024/64=64长度的输入，对应一个输出数字(0,z]
    X[0:64]------->hash()--->z0
    X[64:128]----->hash()--->z1
    X[128:192]---->hash()--->z2
    .....
    
    z0~(0,z],z0,z1...z63就是索引
    
    产生zi是使用kmean算法，也就是说产生产生z个中心(dreduce维度)，
    一共产生m个zi
    
    所以产生的codebook.npy是(m,z,dreduce)维度
    
    :param X: (N,features)
    :param m: 分成m组
    :param z: 每个组输出(0,Z]的数字
    :param outpath: 保存的路径
    :param max_iter: 
    :param start: 
    :return: 
    '''
    import tqdm
    d=X.shape[-1]
    assert d%m==0,"m必须是d 的因数"
    dreduce=d//m
    X=X.reshape((-1,m,dreduce))

    codebook=[0]*(m-start)
    threads=[]

    import multiprocessing
    cpus=multiprocessing.cpu_count()

    for im in tqdm.tqdm(range(start,m)):

        t = MyWorker(im, X[:, im, :], codebook, max_iter)
        threads.append(t)
        t.start()
        if im%cpus==0 and im>0:
            for t in threads:
                t.join()
            threads=[]
    for t in threads:
        t.join()
    codebook = np.array(codebook)
    assert codebook.shape==(m-start,z,dreduce)
    np.save(outpath,codebook)


def loadCodeBook(path='codebook.npy'):
    return np.load(path)
if __name__ == '__main__':

    # pass
    x=np.random.rand(3,512)
    c=np.load('codebook.npy')
    r=quanizer(x,c)
    print(r.shape)
    # #
    # assert r.shape==(N,m)

    # import bcolz
    #
    # x=bcolz.open('/home/zxk/桌面/dump/feature','r')
    # x=x[:].astype(np.float16)
    # print(x.shape)
    # x=np.random.rand(N,d).astype(np.float16)
    # print(x.dtype)
    # createCodeBook(x,m=m,z=z,cpus=-1)