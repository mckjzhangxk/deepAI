import numpy as np

class CoraDataSet():
    '''
    创建了一下重要的属性
        feature:numpy array(float),(V,#features)
        label:numpy array(int),(V,)
        numClass,featurmDims
        A：adjcent matrix,sparse,(V,V)，A[i][j]=1表示j->i的链接存在
        id2Index:字段，节点到A的索引
    '''
    def __init__(self,basepath=None):
        import os
        self.relationfile=os.path.join(basepath,'cora.cites')
        self.featurefile=os.path.join(basepath,'cora.content')
        self._loadDataSet()
    def _strToLabel(self,slabel):
    
        catagory=set(slabel)
        self.numClass=len(catagory)
        class2index={c:i for i,c in enumerate(catagory)}
        r=[class2index[c] for c in slabel]
        return np.array(r)
    def _strToFeature(self,sfeature):
        self.featureDim=sfeature.shape[1]
        r=[]
        for s_list in sfeature:
            r.append([float(x) for x in s_list])
        return np.array(r)

    def _createAdjMatrix(self,ids,graph):
        '''
        返回adjcent matrix,以及 node节点 与 索引 的对于关系
        '''
        id2Index={id:i for i,id in enumerate(ids)}
        from scipy.sparse import csr_matrix
        rows,cols=[],[]
        value=[]
        for tgt,src in graph:
            rows.append(id2Index[tgt])
            cols.append(id2Index[src])
            value.append(1)    
        A=csr_matrix((value,(rows,cols)),shape=(len(ids),len(ids))).tocoo()

        return A,id2Index
    def _loadDataSet(self):
        graph=np.genfromtxt(self.relationfile,dtype=np.str,delimiter='\t')
        feature_label=np.genfromtxt(self.featurefile,dtype=np.str,delimiter='\t')
        
        ids=feature_label[:,0]

        self.A,self.id2Index=self._createAdjMatrix(ids,graph)
        self.feature=self._strToFeature(feature_label[:,1:-1])
        self.label=self._strToLabel(feature_label[:,-1])

        # print(self.A.shape)
        # print(self.feature.shape)
        # print(self.label.shape)
        # print(self.numClass)
    def __repr__(self):
        s='节点数%d\n'%(self.A.shape[0])
        s+='A='+str(self.A.shape)+'\n'
        s+='特征%d\n'%(self.featureDim)
        s+='类别%d\n'%(self.numClass)
        return s     
    def getTorchTensor(self,device='cpu'):
        '''
        调用这个方法得到 pytorch tensor对象
        feature:(N,F) float32
        label:(N,) int64
        A:(N,N),float32 sparse
        '''
        def sparseTensor(X):
            ii=np.stack((X.row,X.col),axis=0)
            indices=torch.from_numpy(ii).long()
            values=torch.from_numpy(X.data).float()
            r=torch.sparse.FloatTensor(indices=indices,values=values,size=X.shape,device='cpu')

            return r.to(device)
        import torch
        label=torch.Tensor(self.label).long().to(device)
        feature=torch.Tensor(self.feature).to(device)
        A=sparseTensor(self.A)       

        # print(feature.shape,feature.dtype,feature.device)
        # print(label.shape,label.dtype,label.device)
        # print(A)
        
        return feature,label,A
if __name__ == "__main__":
    db=CoraDataSet('data/cora')
    print(db)
    # db.getTorchTensor(device='cuda')