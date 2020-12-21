import cv2
import numpy as np
from collections import defaultdict
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve,cg,eigsh
def gauss_pyramid(I):
    ret=[I]
    
    n=int(np.ceil(np.log2(min(I.shape[:2])//16)))
    for i in range(1,n+1):
        ret.append(cv2.pyrDown(ret[i-1]))
    return ret
def laplacian_pyramid(gs):
    ret=[gs[-1]]
    n=len(gs)
    for i in range(n-2,-1,-1):
        g=gs[i]
        H,W=g.shape[:2]
        L=cv2.subtract(g,cv2.pyrUp(gs[i+1],dstsize=(W,H)))
        ret.append(L)
    ret.reverse()
    return ret
def blend_laplician_pyramid(ls_a,ls_b,gs_mask):
    final_la=[]
    for m,la,lb in zip(gs_mask,ls_a,ls_b):
        m=m[:,:,np.newaxis]
        final_la.append(m*la+(1-m)*lb)
    return final_la
def sum_laplacian_pyramid(ls):
    ret=ls[-1]
    n=len(ls)
    for i in range(n-2,-1,-1):
        L=ls[i]
        H,W=L.shape[:2]
        ret=cv2.add(L,cv2.pyrUp(ret,dstsize=(W,H)))
    return ret
def blend(img_a,img_b,mask):
    la_=laplacian_pyramid(gauss_pyramid(img_a))
    lb_=laplacian_pyramid(gauss_pyramid(img_b))
    g_mask=gauss_pyramid(mask)
    return sum_laplacian_pyramid(blend_laplician_pyramid(la_,lb_,g_mask))

def isOMEGA(mask):
    nz=np.nonzero(mask)
    return set(zip(nz[1],nz[0]))

def getBoundary(mask):
    kernel=np.ones((3,3),'int')
    inside=cv2.erode(mask,kernel)
    boundary=cv2.bitwise_xor(mask,inside)
    return isOMEGA(boundary),boundary

def point2VectorIndex(pts):
    return {(x[0],x[1]):i for i,x in enumerate(pts)}

def adj(x,y):
    return [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]

def grid_matrix_param(mask):


    '''

    :param mask:array(H,W) 0/1
    :return:
    data:(x,y,value)
    N:矩阵的大小
    T:key =矩阵的行索引, value=(x,y) 表示邻接点的坐标
    '''
    pts=isOMEGA(mask)
    boundary_pts,_=getBoundary(mask)
    dict_index=point2VectorIndex(pts)

    N=len(pts)
    data=[]
    row=[]
    col=[]
    T=defaultdict(list)



    def f(p):


        pindex=dict_index[p]
        data.append(4.0)
        row.append(pindex)
        col.append(pindex)

        if p not in boundary_pts:
            for q in adj(*p):
                data.append(-1.0)
                row.append(pindex)
                col.append(dict_index[q])
        else:
            for q in adj(*p):
                if q in pts:
                    data.append(-1.0)
                    row.append(pindex)
                    col.append(dict_index[q])

                else:
                    T[pindex].append(q)


    for _ in map(f,pts):pass


    return (data,(row,col)),N,T,dict_index

def dict_index_to_array(data):
    index,xs,ys=[],[],[]
    for pts,i in data.items():
        index.append(i)
        xs.append(pts[0])
        ys.append(pts[1])
    return index,xs,ys


def process(source, target, mask):
    data,N,T,dict_index=grid_matrix_param(mask)
    indexes,xs,ys=dict_index_to_array(dict_index)

    A = csc_matrix(data, dtype=float)

    # Create B matrix
    channels=source.shape[2]
    b = np.zeros((N,channels), dtype=float)

    b[indexes]=source[ys,xs]
    
    for index,pts in T.items():
        for p in pts:
            b[index]+=target[p[1],p[0]]
    
    composite = np.copy(target)
    # x = spsolve(A, b)
    for i in range(channels):
        x=cg(A,b[:,i])
        composite[ys,xs,i]=np.clip(x[0][indexes],0,255)
    return composite

from datetime import datetime
if __name__ == '__main__':
    mask=np.zeros((800,600),'uint8')
    mask[30:130,70:150]=1
    
    src=np.zeros((800,600,3),'uint8')
    target=np.zeros((800,600,3),'uint8')

    # omada=isOMEGA(mask)
    #
    # boundary,boundary_img=getBoundary(mask)
    #
    # for x,y in boundary:
    #     mask[y,x]=128

    # d=point2VectorIndex(omada)
    # print(len(d))
    # print(boundary)


    # data,N,T,dict_index=grid_matrix_param(mask)
    # a,b,c=dict_index_to_array(dict_index)

    # assert N==len(dict_index)
    # for k,v in T.items():
    #     for vv in v:
    #         mask[vv[1],vv[0]]=128
    # cv2.imshow('mask',mask*255)

    # cv2.waitKey(0)
    s=datetime.now()
    sss=process(src,target,mask)
    print(sss.dtype)
    print(datetime.now()-s)