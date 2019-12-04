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

    vp=VPManager(mesh)
    fp=FPManager(mesh)
    pm=PairManager()

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
    
    for vid,q in enumerate(quad):
        vp.setVertexQuadtic(vid,q)

    return quad

class FP:
    def __init__(self,a,b,c):
        self.m_a=a
        self.m_b=b
        self.m_c=c
        self.m_valid=True
    def invalid(self):
        self.m_valid=False
    def getPairs(self):
        if self.m_valid:
            return ((self.m_a,self.m_b),(self.m_b,self.m_c),(self.m_c,self.m_a))
        else:
            return tuple()   
class FPManager():
    def __init__(self,mesh=None):
        self.facelist=[]
        if mesh is not None:
            for f in mesh.f:
                self.add(f[0],f[1],f[2])
    def add(self,a,b,c):
        f=FP(a,b,c)
        self.facelist.append(f)
    def get(self,i):
        return self.facelist[i]
class PairManager:
    def __init__(self):
        self.m_pairid={}
        self.vs=None
        self.fs=None
    def setVertexManager(self,vv):
        self.vs=vv
    def setFaceManager(self,ff):
        self.fs=ff
    def getKey(self,v1,v2):
        key=(v1,v2) if v1<=v2 else (v2,v1)
        return key
    def havePair(self,v1Index,v2Index):
        key=self.getKey(v1Index,v2Index)
        if key in self.m_pairid.keys():
            return True
        else:
            return False
    def addPair(self,v1Index,v2Index):
        v1=self.vs[v1Index]
        v2=self.vs[v2Index]
        if DEBUG:
            assert v1.m_valid and v2.m_valid,"v1,v2 must be valid"
        Q=v1.m_quadtic+v2.m_quadtic
        vv1=v1.m_vertex
        vv2=v2.m_vertex
        
        key=self.getKey(v1Index,v2Index)
        value=vv1.T.dot(Q).vv2

        self.m_pairid[key]=value
    def removePairs(self,pairs):
        for p in pairs:
            if DEBUG:assert len(p)==2
            key=self.getKey(*p)
            if key in self.m_pairid.keys():
                self.m_pairid.pop(key)
            
    def createPairs(self):
        if DEBUG:
            assert self.vs is not None and self.fs is not None
        for vt in self.vs:
            if vt.m_valid:
                vt.createPairs(self.fs,self)      
class VP:
    def __init__(self,v,vid):
        self.m_vertex=v
        self.m_faceids=[]
        self.m_valid=False
        self.m_quadtic=None
        self.vid=vid
    def set_vertex(self,v):
        self.m_vertex=v
    def add_face(self,faceid):
        self.m_faceids.append(faceid)
    def setQuadtic(self,q):
        self.m_quadtic=q
    def getQuadtic(self):
        return self.m_quadtic

    def createPairs(self,facemanager,pairmanager):
        for fid in self.m_faceids:
            face=facemanager.get(fid)
            pairs=face.getPairs()
            for (v1index,v2index) in pairs:
                if v1index!=self.vid and v2index!=self.vid:continue 
                
                if not pairmanager.havePair(v1index,v2index):
                    pairmanager.addPair(v1index,v2index)
            
class VPManager:
    def __init__(self,mesh=None):
        self.m_vslist=[]

        if mesh is not None:
            for vid,v in enumerate(mesh.v):
                self.add(v,vid)
            for faceid,f in enumerate(mesh.f):
                for vid in f:
                    self.m_vslist[vid].add_face(faceid)
            
    def add(self,v,vid):
        vp=VP(v,vid)
        self.m_vslist.append(vp)
    def setVertexFace(self,vid,faceid):
        self.m_vslist[vid].add_face(faceid)
    def setVertexQuadtic(self,vid,q):
        self.m_vslist[vid].setQuadtic(q)

if __name__ == "__main__":
    mesh=loadMesh('data/garg.obj')
    init_quad=init_quadric_per_vertex(mesh)
    # checkSysmetry(init_quad)

    index=4000
    Q=init_quad[index]
    vgood=np.concatenate((mesh.v[index],[1]))

    vselect=miniSovler(Q)
    checkCorrect(Q,vselect,vgood)
