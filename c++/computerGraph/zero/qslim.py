from psbody.mesh import Mesh;
import numpy as np

DEBUG=False

def loadMesh(filename):
    mesh=Mesh(filename=filename)
    print('#vertex %d,#face %d'%(len(mesh.v),len(mesh.f)))
    return mesh
def row(a):
    return np.reshape(a,(1,-1))
def col(a):
    return np.reshape(a,(-1,1))

def miniSovler(Q):
    try:
        checkSysmetry(Q)
        assert Q.shape==(4,4),"Q must be 4x4 matrix"
        D=Q[:3,:3]
        b=Q[:3,3]

        v=np.linalg.solve(D,-b)
        
        return np.concatenate((v,[1]))

    except np.linalg.LinAlgError as e:
        return None

def checkSysmetry(Qs):
    if Qs.ndim==2:
        Qs=[Qs]

    for Q in Qs:
        assert np.all(Q==Q.T)
def checkCorrect(Q,v,vgood):
    print("begin check")
    print(v)
    print(vgood)
    c1=v.T.dot(Q).dot(v)
    c2=v.T.dot(Q).dot(vgood)
    assert np.abs(c1-c2)<1e-8
    print("end check")
def init_quadric_per_vertex(mesh):
    N=len(mesh.v)
    quad=np.zeros((N,4,4))

    for face in mesh.f:
        vs=mesh.v[face]
       
        assert vs.shape==(3,3),"error shape %d,%d"%vs.shape
        H=np.concatenate((vs,np.ones((len(vs),1))),axis=1)

        U,S,VT=np.linalg.svd(H,True)
   
        q=VT[-1]  
        q=q/np.linalg.norm(q[0:3])
        q=col(q)
        if DEBUG:
            assert np.max(np.abs(H.dot(q)))<1e-5

        Q=q.dot(q.T)
        assert Q.shape==(4,4)
        for i in range(3):
            vertexindex=face[i]
            if DEBUG:
                vv=np.concatenate((mesh.v[vertexindex],[1]))
                assert np.abs(vv.T.dot(Q).dot(vv))<1e-8
            quad[vertexindex]+=Q
    return quad

if __name__ == "__main__":
    mesh=loadMesh('data/garg.obj')
    init_quad=init_quadric_per_vertex(mesh)
    # checkSysmetry(init_quad)

    index=4000
    Q=init_quad[index]
    vgood=np.concatenate((mesh.v[index],[1]))

    vselect=miniSovler(Q)
    checkCorrect(Q,vselect,vgood)
