from psbody.mesh import Mesh;
import numpy as np

DEBUG=False

def loadMesh(filename):
    mesh=Mesh(filename=filename)
    print('#from load face:vertex %d,#face %d'%(len(mesh.v),len(mesh.f)))
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
        vt=np.concatenate((v,[1]))
        cost=vt.T.dot(Q).dot(vt)
        return vt,cost
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

    for faceid,face in enumerate(mesh.f):
        vs=mesh.v[face]
       
        assert vs.shape==(3,3),"error shape %d,%d"%vs.shape
        H=np.concatenate((vs,np.ones((len(vs),1))),axis=1)

        U,S,VT=np.linalg.svd(H,True)
   
        q=VT[-1]  
        q=q/np.linalg.norm(q[0:3])
        fp[faceid].set_param(q)
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
    pm.setFaceManager(fp)
    pm.setVertexManager(vp)
    pm.createPairs()
    if DEBUG:
        pm.checkMeshInfo()
    return vp,fp,pm

class FP:
    def __init__(self,a,b,c):
        self.m_a=a
        self.m_b=b
        self.m_c=c
        self.m_valid=True
        self.m_param=None
    def __str__(self):
        return "%d,%d,%d"%(self.m_a,self.m_b,self.m_c)
    def invalid(self):
        self.m_valid=False
    def set_param(self,p):
        self.m_param=p
    def replace(self,vsrc,vtgt):
        if self.m_valid:
            if vsrc==self.m_a:
                self.m_a=vtgt
            elif vsrc==self.m_b:
                self.m_b=vtgt
            elif vsrc==self.m_c:
                self.m_c=vtgt
        if self.m_a==self.m_b or self.m_b==self.m_c or self.m_c==self.m_a:
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
    def __len__(self):
        return len(self.facelist)
    def __iter__(self):
        for mm in self.facelist:
            yield mm
    def __setitem__(self, key, value):
        self.facelist[key]=value
    def __getitem__(self, item):
        return self.facelist[item]

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
    def checkMeshInfo(self):
        c_vs,c_pairs=0,0
        vids=set()
        for vid,v in enumerate(self.vs):
            if v.m_valid:
                c_vs+=1
                vids.add(vid)
        print("#vaid vertex %d"%(c_vs))

        visit_face=set()
        for fid,f in enumerate(self.fs):
            if f.m_valid:
            
                assert (f.m_a in vids) and (f.m_b in vids) and (f.m_c in vids),"face index error,%s"%f
                assert self.getKey(f.m_a,f.m_b,f.m_c) not in visit_face,"duplicate face error %s"%f
                visit_face.add(self.getKey(f.m_a,f.m_b,f.m_c))
        print("#face %d"%(len(visit_face)))

        for key,value in self.m_pairid.items():
            assert (key[0] in vids) and (key[1] in vids),"valid pair index error,(%d,%d)"%key
            c_pairs+=1
        print("#valid pairs %d"%c_pairs)
        print("V-E+F=%d"%(c_vs-c_pairs+len(visit_face)))
    def setVertexManager(self,vv):
        self.vs=vv
    def setFaceManager(self,ff):
        self.fs=ff
    def getKey(self,*args):
        key=sorted(args)
        return tuple(key)
    def havePair(self,v1Index,v2Index):
        key=self.getKey(v1Index,v2Index)
        if key in self.m_pairid.keys():
            return True
        else:
            return False
    def addPair(self,v1Index,v2Index):
        v1=self.vs[v1Index]
        v2=self.vs[v2Index]
        
        assert v1.m_valid and v2.m_valid,"v1,v2 must be valid"
        
        Q=v1.m_quadtic+v2.m_quadtic
        
        
        key=self.getKey(v1Index,v2Index)
        voptimal,cost=miniSovler(Q)
        value=(cost,voptimal)
        if DEBUG:
            if np.abs(cost)<1e-20:
                vvv1=np.concatenate((v1.m_vertex,[1]))
                vvv2 = np.concatenate((v2.m_vertex,[1]))

                c1=vvv1.T.dot(Q).dot(vvv1)
                c2=vvv2.T.dot(Q).dot(vvv2)
                print("create edge (%d,%d) with cost %f"%(v1Index,v2Index,cost))
        self.m_pairid[key]=value
    def removePairs(self,pairs):
        '''
            pairs have form [(v1,v2),(u1,u2),(w1,w2)...]
        '''
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
                pairs=vt.getPairs(self.fs)      
                for (va,vb) in pairs:
                    if not self.havePair(va,vb):
                        self.addPair(va,vb)
    def mergePairs(self,v1Index,v2Index,vt):
        '''
        merge v1Index,v2Index  to vt
        '''
        v1=self.vs[v1Index]
        v2=self.vs[v2Index]
        #first step,remove all pairs in queue associate v1Index,v2Index
        pairs_1=v1.getPairs(self.fs)
        pairs_2=v2.getPairs(self.fs)
    
        self.removePairs(pairs_1)
        self.removePairs(pairs_2)

        #second,update all faces index associate with v1Index,v2Index
        faces_1=v1.m_faceids
        faces_2=v2.m_faceids
        


        newfaces_ids=set()
        newfaces_list=[]

        for faces in [faces_1,faces_2]:
            for faceidx in faces:
                ff=self.fs[faceidx]
                ff.replace(v2Index,v1Index)
                if ff.m_valid:
                    key=self.getKey(ff.m_a,ff.m_b,ff.m_c)
                    if key not in newfaces_ids:
                        newfaces_list.append(faceidx)
                        newfaces_ids.add(key)
     
       
        #third,update 2 vertex
        q1=v1.getQuadtic()
        q2=v2.getQuadtic()
        v1.setQuadtic(q1+q2)
        v1.set_vertex(vt)
        v1.set_facdids(newfaces_list)
        v2.m_valid=False

        #last update compare again using vertex v1
        pairs=v1.getPairs(self.fs)      
        for (va,vb) in pairs:
            if not self.havePair(va,vb):
                self.addPair(va,vb)

    def chiocePairs(self):
        '''
        chioce minimum cost,return v1Index(int),v2Index(int),vt(VP)
        
        '''
        retkey=None
        mincost=float('inf')
        for key,value in self.m_pairid.items():
            c=value[0]
            if c<mincost:
                retkey=key
                mincost=c
        if DEBUG:
            print("merge (%d,%d) with cost %f"%(retkey[0],retkey[1],mincost))
        return retkey[0],retkey[1],self.m_pairid[retkey][1][:3]
    
    def runSlim(self,factor=0.25,iters=None):
        if iters is None:
            N=len(self.vs)
            iters=int(np.ceil(factor*N))
        print(iters)
        for _ in range(iters):
            v1Index,v2Index,vt=self.chiocePairs() 
            self.mergePairs(v1Index,v2Index,vt)
            if DEBUG:
                input("press some key")
                self.checkMeshInfo()

    def export(self,filename):
        vs=[]
        fs=[]

        recordid=0
        dct={}
        for vid,v in enumerate(self.vs):
            if v.m_valid:
                vs.append(np.array(v.m_vertex))
                dct[vid]=recordid
                recordid+=1
        for f in self.fs:
            if f.m_valid:
                a=dct[f.m_a]
                b = dct[f.m_b]
                c = dct[f.m_c]
                fs.append(np.array([a,b,c]))
        vs=np.array(vs).astype(np.float32)
        fs=np.array(fs).astype(np.int32)
        mesh=Mesh(v=vs,f=fs)
        mesh.write_obj(filename=filename)
class VP:
    def __init__(self,v,vid):
        self.m_vertex=v
        self.m_faceids=[]
        self.m_valid=True
        self.m_quadtic=None
        self.vid=vid
    def set_vertex(self,v):
        self.m_vertex=v
    def add_face(self,faceid):
        self.m_faceids.append(faceid)
    def setQuadtic(self,q):
        self.m_quadtic=q
    def set_facdids(self,fs):
        self.m_faceids=fs
    def getQuadtic(self):
        return self.m_quadtic
    '''
    return edge pair associate with this vertex
    '''
    def getPairs(self,facemanager):
        gl=[]
        for fid in self.m_faceids:
            face=facemanager[fid]
            pairs=face.getPairs()
            for (v1index,v2index) in pairs:
                if v1index!=self.vid and v2index!=self.vid:continue 
                gl.append((v1index,v2index))    
        return gl
class VPManager:
    def __init__(self,mesh=None):
        self.m_vslist=[]

        if mesh is not None:
            for vid,v in enumerate(mesh.v):
                self.add(v,vid)
            for faceid,f in enumerate(mesh.f):
                for vid in f:
                    self.m_vslist[vid].add_face(faceid)
    def __iter__(self):
        for mm in self.m_vslist:
            yield mm
    def __len__(self):
        return len(self.m_vslist)
    def __setitem__(self, key, value):
        self.m_vslist[key]=value
    def __getitem__(self, item):
        return self.m_vslist[item]

    def add(self,v,vid):
        vp=VP(v,vid)
        self.m_vslist.append(vp)
    def setVertexFace(self,vid,faceid):
        self.m_vslist[vid].add_face(faceid)
    def setVertexQuadtic(self,vid,q):
        self.m_vslist[vid].setQuadtic(q)

if __name__ == "__main__":
    mesh=loadMesh('a.obj')
    vp,fp,pm=init_quadric_per_vertex(mesh)
    pm.runSlim(factor=0.5)
    pm.export('a.obj')

    # v1=vp[21060]
    # v2=vp[21059]
    # f1=v1.m_faceids
    # f2=v2.m_faceids

    # for fid in f1:
    #     face=fp[fid]
    #     cc1=v1.m_vertex.dot(face.m_param[:3])+face.m_param[3]
    #     cc2=v2.m_vertex.dot(face.m_param[:3]) + face.m_param[3]
    #     print(fid,cc1**2,cc2**2)
    # print("xxxx")
    # for fid in f2:
    #     face = fp[fid]
    #     cc1 = v1.m_vertex.dot(face.m_param[:3]) + face.m_param[3]
    #     cc2 = v2.m_vertex.dot(face.m_param[:3]) + face.m_param[3]
    #     print(fid, cc1**2, cc2**2)
    # pm.runSlim(factor=0.25,iters=2)
    # checkSysmetry(init_quad)
    # index=4000
    # Q=vp[index].getQuadtic()
    # vgood=np.concatenate((vp[index].m_vertex,[1]))
    # vselect,cost=miniSovler(Q)
    # checkCorrect(Q,vselect,vgood)
