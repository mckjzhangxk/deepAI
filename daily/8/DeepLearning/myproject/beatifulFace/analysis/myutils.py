__author__ = 'mathai'
import json
import numpy as np
from collections import defaultdict

array_questions = []
dict_questionId_index = defaultdict(int)
dict_zhuantiId_questions = defaultdict(list)
dict_zsdId_questions = defaultdict(list)
dict_questionId_questions = defaultdict(dict)
dict_zhuantiId_name= defaultdict(str)
dict_all_node=defaultdict(dict)


def normalize(x):
    return x/np.linalg.norm(x)

def randomSigma(martix_cov_half):
    """

    :param martix_cov:协方差矩阵
    :return:z~normal(0,martix_cov),并且单位化
            sigma=sqrt(zT @ martix_cov @z)
    """

    while True:

        za=np.random.randn(martix_cov_half.shape[0])
        z=martix_cov_half.dot(za)
        z=normalize(z)*np.sign(z[0])
        if z[0]>=0 and z[1]>=0:break
    return normalize(za),z,np.sqrt((martix_cov_half.dot(martix_cov_half)).dot(z).dot(z))

def loadChapterData(filename):
    with(open(filename)) as fs:
        chapter = json.load(fs)[0]["children"]

        global array_questions
        global dict_questionId_index
        global dict_zhuantiId_questions
        global dict_zsdId_questions
        global dict_questionId_questions
        global dict_zhuantiId_name
        tmp_zhuantiId_zhd = {}
        stack = []

        for child in chapter:
            dict_all_node[child['id']]=child
            stack.append((child, 1))

        while len(stack) > 0:
            context, level = stack.pop()
            if level == 5:
                dict_questionId_index[context['id']] = len(array_questions)
                array_questions.append([context['nd'], context['zhd'], context['cxd']])
                dict_questionId_questions[context['id']] = context
                continue
            if level == 3:
                _zhuanId = context['id']
                tmp_zhuantiId_zhd[_zhuanId] = context
                dict_zhuantiId_name[_zhuanId]=context['name']

            sub_tasks = context['children']

            for child in sub_tasks:
                dict_all_node[child['id']]=child
                stack.append((child, level + 1))
                if level == 4:
                    _zsdId = context['id']
                    dict_zsdId_questions[_zsdId].append(child['id'])

        for _zhuanId, _zhuanJson in tmp_zhuantiId_zhd.items():
            zsds = _zhuanJson['children']
            for zsd in zsds:
                _zsdId = zsd['id']
                dict_zhuantiId_questions[_zhuanId].extend(dict_zsdId_questions[_zsdId])

        array_questions = np.array(array_questions)


def half_power(M):
    U, S, V = np.linalg.svd(M)

    return U.dot(np.diag(S ** 0.5)).dot(U.T)


def getQuestionInZhuanTi(ztid):
    questionids = dict_zhuantiId_questions[ztid]
    questions = [dict_questionId_questions[q] for q in questionids]
    #
    # for q in questionids:
    # ss=array_questions[dict_questionId_index[q]]
    #     if(np.any(ss>=1)):
    #         print(q)
    question_array = [array_questions[dict_questionId_index[q]] for q in questionids if
                      np.all(array_questions[dict_questionId_index[q]] < 1)]

    return questions, np.array(question_array)
def getQuestionInZsd(zsdid):
    questionids = dict_zsdId_questions[zsdid]
    questions = [dict_questionId_questions[q] for q in questionids]
    #
    # for q in questionids:
    # ss=array_questions[dict_questionId_index[q]]
    #     if(np.any(ss>=1)):
    #         print(q)
    question_array = [array_questions[dict_questionId_index[q]] for q in questionids if
                      np.all(array_questions[dict_questionId_index[q]] < 1)]

    return questions, np.array(question_array)
def getZsdIdByZhuantiId(zhuantiId):
    zhuan_json=dict_all_node[zhuantiId]
    ret=[]

    for x in zhuan_json['children']:
        ret.append(x['id'])
    return ret
def getNodeNameById(nodeId):
    return dict_all_node[nodeId]['name']
def getZhuanTiNameById(ztid):
    return dict_zhuantiId_name[ztid]
def listZhuanTi():
    ret=[]
    for x in dict_zhuantiId_questions.keys():
        ret.append(x)
    return ret
def listZsd():
    ret=[]
    for x in dict_zsdId_questions.keys():
        ret.append(x)
    return ret

def drawArrow(A, direction, ax,color,size):
    '''
    Draws arrow on specified axis from (x, y) to (x + dx, y + dy).
    Uses FancyArrow patch to construct the arrow.

    The resulting arrow is affected by the axes aspect ratio and limits.
    This may produce an arrow whose head is not square with its stem.
    To create an arrow whose head is square with its stem, use annotate() for example:
    Example:
        ax.annotate("", xy=(0.5, 0.5), xytext=(0, 0),
        arrowprops=dict(arrowstyle="->"))
    '''
    # fig = plt.figure()
    #     ax = fig.add_subplot(121)
    # fc: filling color
    # ec: edge color
    ax.arrow(A[0], A[1], direction[0], direction[1],
             length_includes_head=True,  # 增加的长度包含箭头部分
             head_width=size, head_length=size*2, fc='g', ec=color)
    # 注意： 默认显示范围[0,1][0,1],需要单独设置图形范围，以便显示箭头


def analysis(Q):
    u = np.mean(Q, axis=0)
    M = np.cov(Q, rowvar=False)
    sigma = half_power(M)

    U, S, V = np.linalg.svd(M)
    return u, sigma, U[:, 0]*np.sign(U[:, 0][0]), U[:, 1]


def distance_choose(u,Q,direction,sigma,s=0.5,t=0):


    upper=u.dot(direction)-t+s*sigma
    lower=u.dot(direction)-t-s*sigma
    q=Q.dot(direction)

    chioced_set=np.nonzero((q<=upper)*(q>=lower))[0]

    if len(chioced_set)==0:return None
    chioced_index=np.random.choice(chioced_set,1,False)[0]

    return Q[chioced_index]
def distance_choose_minus(u,Q,direction,sigma,s=0.5,t=0):


    upper=u.dot(direction)+t+s*sigma
    lower=u.dot(direction)+t-s*sigma
    q=Q.dot(direction)

    chioced_set=np.nonzero((q<=upper)*(q>=lower))[0]

    if len(chioced_set)==0:return None
    chioced_index=np.random.choice(chioced_set,1,False)[0]

    return Q[chioced_index]
def findGrade(zsdId):
    a=None

    while zsdId in dict_all_node:
        a=dict_all_node[zsdId]
        zsdId=a['parent']
    return a['name']




def methodA_getEdge(nodeA,nodeB,nodeAdj,topK):
    '''
    保留那些 a,b都是彼此邻近的边，返回
    :param nodeA:list
    :param nodeB: list of adjcentNode(node,size=topK)
    :param nodeAdj:list of adjcentNode(node)
    :param topK:
    :return:
    '''
    edges=set()

    for a in nodeA:
        for b in nodeB[a]:
            edges.add((a,b))

    removeList=set()

    for a in nodeA:
        for b in nodeB[a]:
            if (b,a) not  in edges:
                removeList.add((a,b))

    return edges-removeList
def methodA_getEdge(nodeA,nodeB,nodeAdj,topK):
    '''
    保留那些 a,b都是彼此邻近的边，返回
    :param nodeA:list
    :param nodeB: list of adjcentNode(node,size=topK)
    :param nodeAdj:list of adjcentNode(node)
    :param topK:
    :return:
    '''
    edges=set()

    for a in nodeA:
        for b in nodeB[a]:
            edges.add((a,b))

    removeList=set()

    for a in nodeA:
        for b in nodeB[a]:
            if (b,a) not  in edges:
                removeList.add((a,b))

    return edges-removeList

def methodB_getEdge(M,t=0.05):
    nodeA,nodeA_to=np.nonzero(M<=t)
    edges_B=set([(nodeA[i],nodeA_to[i]) for i in range(len(nodeA))])
    return edges_B

def methodC_getEdge(nodeA,nodeB,M,t=0.05):
    edges=set()

    for a in nodeA:
        for b in nodeB[a]:
            if M[a][b]<=t:
                edges.add((a,b))
    return edges

def topK(a,K):
    SI=np.argsort(a,axis=1)
    return SI[:,:K]

def methodD_match(nodeA,nodeAdj,topK):

    candidate=nodeAdj.tolist()
    selected=[[] for i in range(len(candidate))]
    capicity=[topK for i in range(len(candidate))]


    def updateCandidate(node):
        '''
        node 已经完成topK次匹配，所以他不应该出现在任何节点的候选中
        :param node:
        :return:
        '''
        for i in range(len(nodeA)):
            try:
                candidate[i].remove(node)
            except Exception as e:pass
    def addMatchToSelected(node,c):
        '''
        把node,c添加到匹配记录中
        :param node:
        :param c:
        :return:
        '''

        assert c not in selected[node]
        assert node not in selected[c]
        assert c  in candidate[node]
        assert node in candidate[c]
        assert capicity[node]>0
        assert capicity[c]>0

        selected[node].append(c)
        selected[c].append(node)
        candidate[node].remove(c)
        candidate[c].remove(node)


        capicity[node]-=1
        capicity[c]-=1
        if capicity[node]==0:
            updateCandidate(node)
        if capicity[c]==0:
            updateCandidate(c)
    def match(node,c):
        '''
        试图匹配node,c
        这里假设node的最佳候选是c,这里判断 node也是c的候选，我就匹配
        node和c
        :param node:
        :param c:
        :return:
        '''
        ch=candidate[c][:capicity[c]]
        if node in ch:
            addMatchToSelected(node,c)
    def finishMatch(node):
        return capicity[node]==0

    steps=len(candidate[0])
    for k in range(steps):
        for node in nodeA:
            if not finishMatch(node):
                if len(candidate[node])>0:
                    match(node,candidate[node][0])
    for i,adj in enumerate(selected):
        for a in adj:
            assert i in selected[a]

    edges= set([(i,j) for i in range(len(selected)) for j in selected[i]])
    return edges

def featureExtract1(feature):
    feature=np.array(feature)
    M=feature.dot(feature.T)
    np.fill_diagonal(M,-1)
    nodeAdj=topK(-M,-1)
    return -M-np.min(-M),nodeAdj
def featureExtract2(feature):
    feature=np.array(feature)
    A=np.expand_dims(feature,1)
    B=np.expand_dims(feature,0)
    C=np.sqrt((A-B)**2)
    M=np.sum(C,axis=2)
    np.fill_diagonal(M,1000)
    nodeAdj=topK(M,-1)
    return M,nodeAdj

def methodE_match(nodeA,nodeAdj,topK,M,t=0.05):
    edges=methodD_match(nodeA,nodeAdj,topK)
    return set([(a,b) for (a,b) in edges if M[a][b]<=t])
loadChapterData('st.json')

# print(findGrade('9cef2eec2ded11eb99b887cbc3aefd92')['name'])
# listZhuanTi()
# question,question_array=getQuestionInZhuanTi('9cef2eec2ded11eb99b887cbc3aefd92')
# analysis(question_array)
# print(len(question))
# print(question_array.shape)
# for x in getZsdIdByZhuantiId('2a2539bc03e811eb9aaebd621c7eea23'):
#     print(getNodeNameById(x))
# print(len(getZsdIdByZhuantiId('2a2539bc03e811eb9aaebd621c7eea23')))
