from sklearn.cluster import KMeans
import numpy as np
import networkx as nx
from time import  sleep
import json
import glob
import os
from collections import OrderedDict
from utils import *
from utilsHelper import clusterHelper


def printProcess(fn):
    def wrapper(*args,**kwargs):
        print('开始处理:',args[0])
        fn(*args,**kwargs)
        print('结束处理')
    return wrapper

def response(jsonobj,src_file,tgt_dir,labels,cluster_rank,id2Index,colors,pos=None):
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
    :param pos:  每个节点的坐标计算
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
            tokKids=cluster_rank[c]['top'] if c in cluster_rank else [len(nodeId)-1]
            tokKvalues=cluster_rank[c]['value'] if c in cluster_rank else [100.]
            if nodeId in tokKids:
                rank=tokKids.index(nodeId)
                node['topk']=rank
                node['topk_value']=tokKvalues[rank]
            else:
                node['topk']=-1
                node['topk_value']=-1
            color=colors[c]
            node['color']=color

            if pos is not None:
                x,y=pos[index]
                node['x']=config['pad']+int(x)
                node['y']=config['pad']+int(y)
        else:
            node['no_edge']=True
            node['topk']=len(nodeId)-1
            node['topk_value']=100.
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

def computeSpringPosition(G, w=1024, h=768, seed=0):
    '''
    传入图,传出的是numpy 数组
    pos[i]=表示G的节点i的坐标
    :param G: 
    :param w: 
    :param h: 
    :param seed: 
    :return: 
    '''
    pos = nx.spring_layout(G, seed=seed)
    _pos = []
    for p, v in pos.items():
        _pos.append(v)
    pos = np.array(_pos)
    pos_min = np.min(pos, axis=0, keepdims=True)
    pos_max = np.max(pos, axis=0, keepdims=True)
    pos = (pos - pos_min) / (pos_max - pos_min)
    return pos * np.array([[w, h]])

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
        L = LaplacianMatrix(graph2Matrix(G,norm=False))

        ########################老方法#############################################
        # S, V = eig(L,maxK=config['eig_maxtry'],supportDim=config['eig_supportDim'])
        # K = proposalCluster(S, config['proposal_eps'])
        # if K>=len(config['colors']):K=len(config['colors'])
        # labels = getCluster(K, V)
        ########################新方法#############################################
        labels=clusterHelper(L,makK=config['max_clusters'],minNode=config['minNode'],complex=config['max_complex'])

        #######################生成子图后,Page rank###########
        graphs=get_subGraph(G, labels)
        cluster_rank=SubGraphPageRank(graphs,Index2Id,
                         topK=config['pagerank_topk'],
                         alpha=config['pagerank_alpha'],
                         maxiters=config['pagerank_maxiters'],
                         eps=config['pagerank_eps'])
        #####################################################
        if config['canvas_pos']==True:
            pos=computeSpringPosition(G,w=config['canvas_width'],
                                        h=config['canvas_height'],
                                        seed=config['canvas_seed'])
        else:
            pos=None
        #####################################################

        response(jsonObj,
                 src_file=filename,
                 tgt_dir=config['target'],
                 labels=labels,
                 pos=pos,
                 cluster_rank=cluster_rank,
                 id2Index=Id2Index,
                 colors=config['colors'])
    except Exception as e:
        if config['debug']:
            raise e
        saveError(filename, config['target'], e)

if __name__ == '__main__':
    config=loadConfig()
    while True:
        filelist=glob.glob(config['source'])
        if len(filelist)>0:
            for filename in filelist:
                calcRelation(filename,config)
        sleep(config['refresh_interval'])
