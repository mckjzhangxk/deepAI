from sklearn.cluster import KMeans
import numpy as np
import networkx as nx
from time import  sleep
import json
import glob
import os
from collections import OrderedDict,defaultdict


def graph2Matrix(G, indexFunc=None):
    '''
    根据G,返回一个邻接矩阵,A[i][j]表示从node i 到 node j的weight
    indexFunc:索引转换函数,把图G的 "节点名字" 转成 矩阵的 "下标索引"
    如果为None,"节点名字" 就必须是 矩阵的 "下标索引"
    '''
    n = len(G.nodes)
    A = np.zeros((n, n))
    for nodeIdx, adj in G.adjacency():
        for adjIdx, w in adj.items():
            s, t = (indexFunc[nodeIdx], indexFunc[adjIdx]) if indexFunc else (nodeIdx, adjIdx)
            A[s, t] = w['weight'] if 'weight' in w else 1
    return A


def LaplacianMatrix(A):
    '''
    传入邻接矩阵,A[i][j]表示从node i 到 node j的权重
    计算degress D,返回L=D-A
    '''
    D = np.diag(A.sum(axis=1))
    return D - A


def eig(L):
    '''
    计算L的特征值和特征向量,并按照升序排列
    返回S:r特征值
       V:n,r特征向量
    '''
    S, V = np.linalg.eig(L)
    sortIndex = np.argsort(S)
    S = S[sortIndex].real
    V = V[:, sortIndex].real
    return S, V


def proposalCluster(S, eps=0.2):
    '''

        eps:当某个特征值与他临近的差值大于eps时,
        这个特征值就是一个分界点,特征值的索引+1表示应该分类
        的个数

        返回建议分的类别,并且取出相应的
        特征,进行返回K


    '''

    K = 1 + np.where(S < eps)[0][-1]
    return K


def getCluster(K, eigValues):
    '''
    提取特征eigValues[:,1:K],然后运行分类算法
    '''
    features = eigValues[:, 1:K]
    model = KMeans(n_clusters=K)
    model.fit(features)
    return model.labels_


def makeChengfGraph(filepath):
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
        sid = Id2Index[s]
        tid = Id2Index[t]
        G.add_edge(sid, tid, weight=1.0)
        edgenum += 1

    print('#全部节点数量,有临边节点数量,#边数量(双向):', len(nodes),len(Id2Index),edgenum)

    return G, Id2Index, Index2Id,jsonObj





def get_subGraph(G, labels):
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

    # clusters:key=classname,value=list of edges(s,t,w)
    clutsers = defaultdict(list)
    for n1, adj in G.adjacency():
        for n2, w in adj.items():
            c1, c2 = labels[n1], labels[n2]
            if c1 != c2: continue
            weight = w['weight'] if 'weight' in w else 1.0
            clutsers[c1].append((n1, n2, weight))

    # 为每个"群"构建一张新图
    graphs = {}
    for k, cluster_edges in clutsers.items():
        G = nx.Graph()
        for e in cluster_edges:
            G.add_edge(e[0], e[1], weight=e[2])
        G_Node_2Index = {n: i for i, n in enumerate(G.nodes)}
        G_Node_Index2 = {v: k for k, v in G_Node_2Index.items()}
        graphs[k] = {'graph': G,
                     'index2': G_Node_Index2,
                     '2index': G_Node_2Index
                     }

        print('聚类:{},节点数量:{},边数量(无向):{}'.format(k, len(G.node), len(G.edges)))
    return graphs


def pageRank(M, alpha=0.85, eps=0.1, maxiters=300):
    '''
        计算每个节点的 "价值",要求

        M:概率转换矩阵M[i,j]表示节点j到节点i的概率,sum(M[:,j])=1 

        V_t=alpha+(1-alpha)*M*V_t-1

        返回:降序返回
            V,表示每个节点的 "价值",
            rank:排名
    '''

    assert np.all(np.abs(np.sum(M, axis=0) - 1) < 1e-2), "矩阵M 的一列 之和 必须 为 1"

    t = 0

    V = np.ones(M.shape[0]) / M.shape[0]
    while t < maxiters:
        V_old = V
        V = alpha + (1 - alpha) * M.dot(V_old)
        if np.max(np.abs(V - V_old)) < eps: break
        t += 1
    sort_index = np.argsort(-V)
    V = V[sort_index]
    return V, sort_index


def SubGraphPageRank(subgraphs, global_name_fn, topK=3, alpha=0.85, maxiters=300, eps=0.01):
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
    cluster_rank = {}

    for c, graphobj in subgraphs.items():
        g, g_2index, g_index2 = graphobj['graph'], graphobj['2index'], graphobj['index2']

        adjM = graph2Matrix(g, g_2index)
        ###临接矩阵 转化 概率 转化 矩阵###################
        # 注意1 要对 "一行" 归一化,因为一行表示 到 其他 的转换
        # 注意2 要 "转置" adjcent Matrix(如果是非对称),因为adjcent Matrix "一行" 表示  到其他 节点
        # 的转换,而 page rank 要求 "一列"  表示 到 其他节点 的转换,
        P = adjM / adjM.sum(axis=1, keepdims=True)
        P = P.T
        #######################################

        V, rank = pageRank(P, alpha=alpha, eps=eps, maxiters=maxiters)

        _topK = min(topK,V.shape[0]) if topK > 0 else V.shape[0]

        top_value = V[:_topK]
        top_global_name = [global_name_fn[g_index2[rank[i]]] for i in range(_topK)]
        top_gloabl_index = [g_index2[rank[i]] for i in range(_topK)]
        cluster_rank[c] = {'top': top_global_name,
                           'top_index': top_gloabl_index,
                           'value': top_value}
    return cluster_rank

def printProcess(fn):

    def wrapper(*args,**kwargs):
        print('开始处理:',args[0])
        fn(*args,**kwargs)
        print('结束处理')
    return wrapper

def response(jsonobj,src_file,tgt_dir,labels,cluster_rank,id2Index,colors):
    '''
    把分类的结果写入到tgt_dir下面,删除src_file,
    修改jsonobj对象的color属性
    
    对jsonobj 更新
        根据分类结果,更新color属性
        根据排名添加topk,topk-value属性
        对于没有边的节点,添加no_edge=true属性
        成功处理后,设置success=true属性
    :param jsonobj:从src_file读取的json 对象 
    :param src_file: 更新结束后要删除
    :param tgt_dir: 输出的 文件
    :param labels:  分类结果
    :param id2Index: 节点的名字到索引的映射
    :param id2Index: 索引到节点名字的映射
    :return: 
    '''
    nodelist=jsonobj['nodes']

    for node in nodelist:
        nodeId=node['id']

        if nodeId in id2Index:
            index=id2Index[nodeId]
            c=labels[index]
            #本类的topK
            tokKids=cluster_rank[c]['top']
            tokKvalues=cluster_rank[c]['value']
            if nodeId in tokKids:
                rank=tokKids.index(nodeId)
                node['topk']=rank
                node['topk-value']=tokKvalues[rank]
            else:
                node['topk']=-1
                node['topk-value']=-1
            color=colors[c]
            node['color']=color
        else:
            node['no_edge']=True
            print('节点%s,是孤立节点'%nodeId)
    ############同步文件,删除源文件,输出目标文件##################
    if os.path.exists(src_file):
        os.remove(src_file)
    tgt_file=os.path.join(tgt_dir,os.path.basename(src_file))
    with open(tgt_file,'w') as fp:
        jsonobj['success']=True
        json.dump(jsonobj,fp,indent=1)
    #########################################################
def loadConfig():
    with open('config.json') as fp:
        config = json.load(fp)
    return config
def saveError(src, tgt_dir, err):
    '''
    保存错误信息到tgt_dir,然后清除源文件
    :param src: 源请求json文件
    :param tgt_dir: 
    :param err: 错误信息
    :return: 
    '''
    if os.path.exists(src):
        with open(src) as fp:
            errobj = json.load(fp)
    else:
        errobj=OrderedDict()
    errobj['success']=False
    errobj['file']=src
    errobj['info']=str(err)

    tgt_file = os.path.join(tgt_dir, os.path.basename(src))
    with open(tgt_file, 'w') as fp:
        json.dump(errobj, fp, indent=1)
    if os.path.exists(src):
        os.remove(src)
@printProcess
def calcRelation(filename,config):
    '''
    开始处理 请求,
    1)聚类
    2)根据类别分割子图
    3)子图排名
    4)响应请求
    :param filename: 
    :param config: 
    :return: 
    '''
    try:
        G, Id2Index, Index2Id, jsonObj = makeChengfGraph(filename)
        #############根据临界矩阵进行谱分类######################
        L = LaplacianMatrix(graph2Matrix(G))
        S, V = eig(L)
        K = proposalCluster(S, config['proposal_eps'])
        labels = getCluster(K, V)

        #######################生成子图后,Page rank###########
        graphs=get_subGraph(G, labels)
        cluster_rank=SubGraphPageRank(graphs,Index2Id,
                         topK=config['pagerank_topk'],
                         alpha=config['pagerank_alpha'],
                         maxiters=config['pagerank_matiters'],
                         eps=config['pagerank_eps'])

        response(jsonObj,
                 src_file=filename,
                 tgt_dir=config['target'],
                 labels=labels,
                 cluster_rank=cluster_rank,
                 id2Index=Id2Index,
                 colors=config['colors'])
    except Exception as e:
        saveError(filename, config['target'], e)
if __name__ == '__main__':
    config=loadConfig()
    while True:
        filelist=glob.glob(config['source'])
        if len(filelist)>0:
            for filename in filelist:
                calcRelation(filename,config)
        sleep(config['refresh_interval'])