import numpy as np

def general_iou(A,B):
    '''
    calc iou between A and B
    A may have shape (D1,D2,...Dk,4),
    B may have shape (d1,d2,...dj,4)
    return value have shape
    (D1,D2,...Dk,d1,d2,...dj)
    '''
    if isinstance(A,list) or isinstance(A,tuple):
        A=np.array(A)
    if isinstance(B,list) or isinstance(B,tuple):
        B=np.array(B)

    Ashape=A.shape
    Bshape=B.shape

    
    assert Ashape[-1]==Bshape[-1]==2 or Ashape[-1]==Bshape[-1]==4,'last dim must have length 4'
    dfeature=Ashape[-1]

    dimA=len(Ashape)
    dimB=len(Bshape)
    for i in range(dimB-1):
        A=np.expand_dims(A,axis=dimA-1)

    for i in range(dimA-1):
        B=np.expand_dims(B,axis=0)
    #(DA,DB,2)
    if dfeature==4:
        left=np.maximum(A[...,0:2],B[...,0:2])
        right=np.minimum(A[...,2:4],B[...,2:4])
        AreaA=(A[...,2]-A[...,0])*(A[...,3]-A[...,1])
        AreaB=(B[...,2]-B[...,0])*(B[...,3]-B[...,1])
    else:
        right=np.minimum(A[...,0:2],B[...,0:2])
        left=np.zeros_like(right)
        
        AreaA=A[...,0]*A[...,1]
        AreaB=B[...,0]*B[...,1]
    #(DA,DB)
    w=np.maximum(right[...,1]-left[...,1],0)
    h=np.maximum(right[...,0]-left[...,0],0)
    intersection=w*h

    iou=intersection/(AreaA+AreaB-intersection)

    return iou

# A=np.random.rand(2)
# print(A)
# B=np.random.rand(2,2)
# print(B)
# iou=general_iou(A,B)
# print(iou.shape)
# print(iou)
