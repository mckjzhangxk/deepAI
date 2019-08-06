from utils import *
import numpy as np

def pickBest(queue):
    maxsofar=-1
    popindex=-1
    for i,(L,indexes), in enumerate(queue):
        cm=np.diag(L).sum()/2/len(L)
        if cm>maxsofar:
            popindex=i
            maxsofar=cm
            # maxsofar=len(indexes)

    return queue.pop(popindex)
def getPairCluster(L):
    def updateDiag(l):
        lsum=l.sum(axis=1)
        i1,i2=np.diag_indices_from(l)
        l[i1,i2]-=lsum
    S,V=eig(L)
    label=V[:,1]>0
    c1=np.where(label)[0]
    c2=np.where(1-label)[0]
    L1=L[c1][:,c1]
    L2=L[c2][:,c2]


    updateDiag(L1)
    updateDiag(L2)
    assert np.all(L1.sum(axis=1)==0)
    assert np.all(L2.sum(axis=1) == 0)
    return L1,L2,c1,c2

def clusterHelper(L,makK=16,minNode=5,complex=1.8):
    def initqueue(L):
        S,V=eig(L)
        disconnected=(S<1e-8).sum()
        if disconnected>1:
            labels=getCluster(disconnected,V)
            ret=[]
            for k in range(disconnected):
                ii=np.where(labels==k)[0]
                if len(ii)>0:
                    ret.append((L[ii][:,ii],ii))
            return ret
        
        else:
            return [(L,np.arange(len(L)))]
    N=len(L)
    queue=initqueue(L)
    initLen=len(queue)

    for k in range(makK-initLen):
        L,sp=pickBest(queue)
        cm=np.diag(L).sum()/2/len(L)
        print(len(L),cm)
        if cm<complex:
            queue.append((L,sp))
            break
        if len(sp)<minNode:break

        L1,L2,c1,c2=getPairCluster(L)
        if len(c1)>0:
            queue.append((L1,sp[c1]))
        if len(c2)>0:
            queue.append((L2,sp[c2]))
    labels=np.zeros(N)-1

    for i,(_,lab) in enumerate(queue):
        labels[lab]=i
    return labels.astype(np.int32)
# import numpy as np
# if __name__ == '__main__':
#     # a=np.random.rand(10,10)
#     # getPairCluster(a)
#     G, Id2Index, Index2Id, _ = makeChengfGraph('resbak/15644536933815.json')
#     L = LaplacianMatrix(graph2Matrix(G, norm=False))
#     labels = clusterHelper(L)
#
#     # print(np.histogram(labels))
#     print(np.unique(labels))