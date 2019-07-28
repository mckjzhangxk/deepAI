import numpy as np

def iou(A,B):
    '''
    x1,y1,x2,y2 form
    union(A,B)/A
    '''

    if isinstance(A,list):
        A=np.array(A)
    if isinstance(B,list):
        B=np.array(B)

    for _ in range(B.ndim-1):
        A=np.expand_dims(A,axis=-2)
    B=np.expand_dims(B,0)
    
    xleft =np.maximum(A[...,0],B[...,0])
    xright=np.minimum(A[...,2],B[...,2])

    yleft=np.maximum(A[...,1],B[...,1])
    yright=np.minimum(A[...,3],B[...,3])

    union=np.maximum(0,(xright-xleft)*(yright-yleft))
    areaA=np.maximum(0.1,(A[...,2]-A[...,0])*(A[...,3]-A[...,1]))
    areaB=np.maximum(0.1,(B[...,2]-B[...,0])*(B[...,3]-B[...,1]))
    r=union/(areaA+areaB)

    return r

def ioa(A,B):
    '''
    x1,y1,x2,y2 form
    union(A,B)/A
    '''

    if isinstance(A,list):
        A=np.array(A)
    if isinstance(B,list):
        B=np.array(B)

    for _ in range(B.ndim-1):
        A=np.expand_dims(A,axis=-2)
    B=np.expand_dims(B,0)
    
    xleft =np.maximum(A[...,0],B[...,0])
    xright=np.minimum(A[...,2],B[...,2])

    yleft=np.maximum(A[...,1],B[...,1])
    yright=np.minimum(A[...,3],B[...,3])

    W=np.maximum(xright-xleft,0)
    H=np.maximum(yright-yleft,0)
    union=W*H
    areaA=np.maximum(0.1,(A[...,2]-A[...,0])*(A[...,3]-A[...,1]))
    r=union/areaA

    return r

if __name__ == "__main__":
    face_box=np.random.rand(10,4)
    face_box[:,2:4]=face_box[:,0:2]+1
    person_box=np.random.rand(3,4)
    person_box[:,2:4]=person_box[:,0:2]+1

    xx=ioa(face_box,person_box)
    print(xx)