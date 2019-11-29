import numpy as np


def isBiTonic(a):
    N=len(a)

    for i in range(1,N//2):
        if a[i]<a[i-1]:return False
    for i in range(1+N//2,N):
        if a[i]>a[i-1]:return False
    return True

def createBiTonic(N=8,T=100):
    start=0
    ret=[]
    for i in range(0,N//2):
        x=np.random.randint(start,T)
        ret.append(x)
        start=x+1
    start=T
    for i in range(N//2,N):
        x=np.random.randint(0,start)
        ret.append(x)
        start=x-1
    assert len(ret)==N
    assert isBiTonic(ret)
    return ret
def biTonicSplit(a):
    N=len(a)
    x1=a[:N//2]
    x2=a[N//2:]
    y1,y2=[],[]

    for i in range(N//2):
        cmin=min(x1[i],x2[i])
        cmax=max(x1[i],x2[i])
        y1.append(cmin)
        y2.append(cmax)
    print(y1)
    print(y2)
    assert len(y1)==(N//2) and len(y2)==(N//2)
    assert isBiTonic(y1) and isBiTonic(y2)
    return (y1,y2)
if __name__ == "__main__":
    # r=createBiTonic(16,1000)
    # print(r)
    r=[19, 756, 869, 932, 964, 965, 996, 999, 436, 375, 232, 30, 11, 8, 3, 1]

    assert(isBiTonic(r))
    biTonicSplit(r)
