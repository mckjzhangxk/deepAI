from sklearn.cluster import KMeans
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import scipy.sparse as sparse
from collections import defaultdict

def isSparse(A):
    return (np.count_nonzero(A)/A.size)<0.1
def isSysmetric(A):
    return np.all(A==A.T)
def makeChengfGraph(filepath):

    '''
    生成的图 节点 名称 是索引 而不是原始 id
    :param filepath: 
    :return: 
    '''
    with open(filepath) as fp:
        jsonObj = json.load(fp)
    nodes = []
    for node in jsonObj['nodes']:
        pid = node['id']
        nodes.append(pid)

    # 创建关系图
    nodename = set()
    for edge in jsonObj['edges']:
        s = edge['source']
        t = edge['target']

        nodename.add(s)
        nodename.add(t)

    Id2Index = {phone: i for i, phone in enumerate(nodename)}
    Index2Id = {v: k for k, v in Id2Index.items()}

    G = nx.Graph()
    edgenum = 0
    for edge in jsonObj['edges']:
        s = edge['source']
        t = edge['target']
        weight=edge['size'] if 'size' in edge else 1.0
        weight=weight if weight>0 else 1.0

        weight=1.0

        # print('xxxxxxxxxxxxxxxx',weight)
        sid = Id2Index[s]
        tid = Id2Index[t]
        G.add_edge(sid, tid, weight=weight)
        edgenum += 1

    print('#全部节点数量,有临边节点数量,#边数量(双向):', len(nodes),len(Id2Index),edgenum)

    return G, Id2Index, Index2Id,jsonObj

def graph2Matrix(G,indexFunc=None,norm=True,supportSparse=False):
    '''
    根据G,返回一个邻接矩阵,A[i][j]表示从node i 到 node j的weight
    indexFunc:索引转换函数,把图G的 "节点名字" 转成 矩阵的 "下标索引"
    如果为None,"节点名字" 就必须是 矩阵的 "下标索引"
    
    如果norm=True,矩阵的一行之和 是1,
    
    如果G是稀疏图,那么会返回稀疏矩阵
    '''
    n=len(G.nodes)
    A=np.zeros((n,n))
    for nodeIdx,adj in G.adjacency():
        for adjIdx,w in adj.items():
            s,t= (indexFunc[nodeIdx],indexFunc[adjIdx]) if indexFunc else (nodeIdx,adjIdx)    
            if s!=t:
                A[s,t]=1 #w['weight'] if 'weight' in w else 1
    if norm:
        A=A/np.sum(A,axis=1,keepdims=True)
    if supportSparse and isSparse(A):
        A=sparse.csr_matrix(A)
        print('使用稀疏表示法')
    return A
def LaplacianMatrix(A):
    '''
    传入邻接矩阵,A[i][j]表示从node i 到 node j的权重
    计算degress D,返回L=D-A
    
    传入的A可以是dense or sparse,传出不改变矩阵类型
    '''
    if isinstance(A,np.ndarray):
        D=np.diag(A.sum(axis=1))
        return D-A
    else:
        d=np.asarray(A.sum(axis=1)).ravel()
        D=sparse.diags(d)
        return D-A
def eig(L,maxK=30,maxTry=5,supportDim=500):
    '''
    计算L的 "头maxK个" 特征值和特征向量,并按照升序排列,
    计算eig时候,需要给定ncv
        ncv越大,算法越慢,但是成功率越高
        ncv小,算法快,但可能失败,所有要有重试maxTry
        对于dim(L)<=supportDim的矩阵,ncv就是dim(L)
        否则是自动设在(:
    
    返回S:r特征值
       V:n,r特征向量
    '''
    n=L.shape[0]

    ncv=None if n>supportDim else n
    ###########老方法###########################################
#     for i in range(maxTry):
#         try:
# #             S,V=sparse.linalg.eigs(L,k=min(maxK,n-2),which='SM',ncv=ncv)
# #             S,V=np.linalg.eig(L)
#             break
#         except sparse.linalg.ArpackError as e:
#             print(i,e)
#             if i==maxTry-1:
#                 raise e
    ######################################################
    V,S,_=np.linalg.svd(L,full_matrices=False)
    S=S.real
    V=V.real
    ii=np.argsort(S)
    S=S[ii]
    V=V[:,ii]
    return S,V
def proposalCluster(S,eps=0.2):
    '''
    
        eps:当某个特征值与他临近的差值大于eps时,
        这个特征值就是一个分界点,特征值的索引+1表示应该分类
        的个数
        
        返回建议分的类别,并且取出相应的
        特征,进行返回K
        
        
    '''
    
    K=1+np.where(S<eps)[0][-1]
    return K
def getCluster(K,eigValues):
    '''
    提取特征eigValues[:,1:K],然后运行分类算法
    '''
    if K==1:K=2
    features=eigValues[:,1:K]
    model=KMeans(n_clusters=K,n_jobs=-1)
    model.fit(features)
    return model.labels_

def get_subGraph(G,labels):
    '''
    G:要求节点名称是0开始的索引,与labels相对于,也就是说
    labels[node'i name]表示node i的分类
    
    labels:G节点 聚类的结果,numpy array
    
    返回:cluster=dict
    key:分组名称
    value:dict{
        graph:构建的子图,子图节点使用的名字继承自父图G
        2index:子图 "节点名称" 与 "子图节点" 的对应字典,全局 到 局部 ,名字 到 索引
        index2:局部 到 全局, 索引 到 名字,
    }
    '''
    
    #clusters:key=classname,value=list of edges(s,t,w)
    clutsers=defaultdict(list)
    for n1,adj in G.adjacency():
        for n2,w in adj.items():
            c1,c2=labels[n1],labels[n2]
            if c1!=c2:continue
            weight=w['weight'] if 'weight' in w else 1.0
            clutsers[c1].append((n1,n2,weight))
    
    #为每个"群"构建一张新图
    graphs={}
    for k,cluster_edges in clutsers.items():
        G=nx.Graph()
        for e in cluster_edges:
            G.add_edge(e[0],e[1],weight=e[2])
        G_Node_2Index={n:i for i,n in enumerate(G.nodes)}
        G_Node_Index2={v:k for k,v in G_Node_2Index.items()}
        graphs[k]={'graph':G,
                   'index2':G_Node_Index2,
                   '2index':G_Node_2Index
                  }
        
        print('聚类:{},节点数量:{},边数量(无向):{}'.format(k,len(G.node),len(G.edges)))
    return graphs

def pageRank(M,alpha=0.85,eps=0.1,maxiters=300):
    '''
        计算每个节点的 "价值",要求
        
        M:概率转换矩阵M[i,j]表示节点j到节点i的概率,sum(M[:,j])=1 
        
        V_t=alpha+(1-alpha)*M*V_t-1
        
        返回:降序返回
            V,表示每个节点的 "价值",
            rank:排名
    '''
    
    assert np.all(np.abs(np.sum(M,axis=0)-1)<1e-2),"矩阵M 的一列 之和 必须 为 1"
    
    t=0
    
    V=np.ones(M.shape[0])/M.shape[0]
    while t<maxiters:
        V_old=V
        if isinstance(M,np.ndarray):
            V=alpha+(1-alpha)*M.dot(V_old)
        else:
            p=M.dot(sparse.csr_matrix(V_old).T).toarray().ravel()
            V=alpha+(1-alpha)*p
        if np.max(np.abs(V-V_old))<eps:break
        t+=1
    sort_index=np.argsort(-V)
    V=V[sort_index]
    return V,sort_index
def SubGraphPageRank(subgraphs,global_name_fn,topK=3,alpha=0.85,maxiters=300,eps=0.01):
    '''
    综合的函数,对每个 "子图" 进行 pagerank,选出 topK,后返回
    这个节点 的全局 名字,以及 在子图 的 "价值"
    
    subgraphs:dict of subgraph dict
        subgraph:有graph,和index属性
        graph是子图对象
        2index:是 "全局节点索引"  转换到 子图 "节点索引" 的函数,全局索引 是子图节点 的名字,全局 转 局部
        index2:局部 转 全局
        
    global_name_fn:全局索引 向 全局名称 的转化 函数
    topK:对每个 类 选择topK,
    alpha,maxiters,eps:分别对于pagerank的参数
    
    返回:
    cluster_rank:
        key=聚类的名称
        value=聚类的统计(rank结果),dict
            top:list of topK 节点的全局名称
            value:list of topK 节点 的 "价值"(相对于子图)
    '''
    cluster_rank={}
    
    for c,graphobj in subgraphs.items():
        g,g_2index,g_index2=graphobj['graph'],graphobj['2index'],graphobj['index2']
        
        adjM=graph2Matrix(g,g_2index,supportSparse=True)
        ###临接矩阵 转化 概率 转化 矩阵###################
        #注意1 要对 "一行" 归一化,因为一行表示 到 其他 的转换
        #注意2 要 "转置" adjcent Matrix(如果是非对称),因为adjcent Matrix "一行" 表示  到其他 节点
        #的转换,而 page rank 要求 "一列"  表示 到 其他节点 的转换,

        P=adjM.T

        #######################################
        
        V,rank=pageRank(P,alpha=alpha,eps=eps,maxiters=maxiters)
        K=topK if topK>0 else len(V)
        top_value=V[:K]
        top_global_name=[global_name_fn[g_index2[rank[i]]] for i in range(K)]
        top_gloabl_index=[g_index2[rank[i]] for i in range(K)]
        cluster_rank[c]={'top':top_global_name,
                         'top_index':top_gloabl_index,
                         'value':top_value}
    return cluster_rank


def computeLayout(subgraph, seed, iterations=100):
    H = nx.Graph()
    for c, sub in subgraph.items():
        H.add_edges_from(sub['graph'].edges())
    pos = nx.spring_layout(H, seed=seed, iterations=iterations)
    pos = np.array([pos[i] for i in range(len(pos))])

    pos_min = np.min(pos, axis=0, keepdims=True)
    pos_max = np.max(pos, axis=0, keepdims=True)
    pos = (pos - pos_min) / (pos_max - pos_min)

    return pos


def basicLayout(jsondata):
    allnodes = [[node['x'], node['y']] for node in jsondata['nodes']]
    allnodes = np.array(allnodes)
    return allnodes

def scalePosition(pos, label, factor=1.0):
    def aa(P):
        center = P.mean(axis=0, keepdims=True)
        P_center = P - center
        M = P_center.T.dot(P_center)
        U, S, V = np.linalg.svd(M, False)
        u1, u2 = U[:, 0], U[:, 1]
        Q = np.array([u1, u2])

        #         diag=factor*np.diag([S[0],S[1]])
        diag = np.eye(2)
        diag[0, 0] = S[1] / S[0]

        r = P_center.dot(Q).dot(factor * diag).dot(Q.T)
        r = r + center
        return r

    pos = np.array([pos[i] for i in range(len(pos))])
    pos = pos.copy()
    clusters = np.unique(label)
    for k in clusters:
        mask = (label == k)
        cls_k = pos[mask]

        pos[mask] = aa(cls_k)
    return pos
def convert(pos, ref):
    '''
    pos:0,1 scale (N',2)
    ref:unknown scale (N,2)
    '''

    ref_min = np.min(ref, axis=0, keepdims=True)
    ref_max = np.max(ref, axis=0, keepdims=True)
    ref_wh = ref_max - ref_min

    # r = pos * 1000 + ref_min
    minxy=pos.min(axis=0)
    r=(pos-minxy+0.1)*5000
    return r


def computeSpringPosition(subgraph,jdata, label,seed=0, layout_iters=100):
    refs = basicLayout(jdata)
    pos = computeLayout(subgraph, seed, layout_iters)
    pos=scalePosition(pos,label,factor=4.0)
    pos = convert(pos, refs)
    return pos