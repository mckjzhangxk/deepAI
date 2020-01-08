import networkx as nx
import numpy as np

class GraphWrapper():
    def __init__(self,G,weight='weight'):
        self.G=G
        self.weight=weight
        self.m2=None
    def getNodes(self):
        return [n for n in self.G.nodes()]
    def getNodeDegree(self,i):
        return self.G.degree(i,weight=self.weight)
    def getPairWeight(self,i,j):
        if not self.G.has_edge(i,j):return 0
        return self.G.get_edge_data(i,j)[self.weight]
    def getTotalWeight(self):
        if self.m2 is not None:
            return self.m2
        r=0
        for edge in self.G.edges():
            r+=self.G.get_edge_data(edge[0],edge[1])[self.weight]
        self.m2=r*2
        return self.m2

    def getNodeNeibours(self,i):
        return {key:val[self.weight] for key,val in self.G[i].items()}

    def getNodeWeightBetweenSet(self,nodei,S):
        r=0
        w_dict=self.getNodeNeibours(nodei)
        for s in S:
            if s in w_dict:
                r+=w_dict[s]
        return 2*r

    def newGraph(self,edges):
        G = nx.Graph()
        for (e1,e2),w in edges.items():
            G.add_edge(e1,e2,weight=w)
        return GraphWrapper(G,self.weight)
class Community():
    def __init__(self, G, node):
        '''
        输入G,node表示一个节点构成一个community

        '''
        self.G=G
        self.sigma_in = 2*G.getPairWeight(node,node)
        self.sigma_tot = G.getNodeDegree(node)
        self.w = self.sigma_tot - self.sigma_in
        assert self.w >= 0
        self.members = [node]
        # self.degrees = [G.degree(n, weight=weight) for n in nodes]
    def __len__(self):
        return len(self.members)
    def empty(self):
        return len(self.members)==0

    def has_node(self,n):
        return n in self.members

    def modularity_add(self,i):
        assert not self.has_node(i)
        m_2 = self.G.getTotalWeight()
        ki =self.G.getNodeDegree(i)

        k_i_in =self.G.getNodeWeightBetweenSet(i,self.members+[i])

        r = k_i_in / m_2 - (2 * ki * self.sigma_tot) / (m_2 ** 2)
        return r, k_i_in

    def modularity_remove(self,i):
        assert  self.has_node(i)

        m_2 = self.G.getTotalWeight()
        ki = self.G.getNodeDegree(i)

        k_i_out = self.G.getNodeWeightBetweenSet(i,self.members)

        r = -k_i_out / m_2 + (2 * ki * self.sigma_tot - 2 * ki ** 2) / (m_2 ** 2)
        return r, k_i_out

    def merge(self, other, nodei,ki, K_i_in, K_i_out):

        '''
        
        :param other: 另外一个community
        :param nodei: other里面的节点
        :param ki: nodei的度
        :param K_i_in: nodei加入本community会带来的权重提升
        :param K_i_out:nodei退出本other'scommunity会带来的权重减少 
        :return: 
        '''
        assert other.has_node(nodei)

        self.sigma_tot += ki
        self.sigma_in += K_i_in

        other.sigma_tot -= ki
        other.sigma_in -= K_i_out

        self.w = self.sigma_tot - self.sigma_in
        assert self.w >= 0

        self.members.append(nodei)
        other.members.remove(nodei)


class Louvain():
    def __init__(self, G):
        self.G = G
        self.nodes = G.getNodes()
        self.cms = []
        self.node_community = []

        for i, node in enumerate(self.nodes):
            self.cms.append(Community(G,node))
            self.node_community.append(i)

    def run_one_level(self):
        init_community=self.node_community[:]
        old_community = self.node_community[:]
        while True:
            rnd = np.random.permutation(self.nodes)
            for node in rnd:
                ci = self.node_community[node]
                C = self.cms[ci]
                ki = self.G.getNodeDegree(node)
                leave_Q, k_out = C.modularity_remove(node)

                dQmax ,chioce,k_in= 0,-1,-1


                for nodej in self.G.getNodeNeibours(node):
                    cj=self.node_community[nodej]
                    if cj==ci:continue
                    enter_Q,_kin=self.cms[cj].modularity_add(node)
                    dQ = enter_Q + leave_Q
                    if dQ > dQmax:
                        dQmax = dQ
                        chioce = cj
                        k_in = _kin

                if chioce != -1 and chioce != ci:
                    self.cms[chioce].merge(C,node,ki,k_in, k_out)
                    self.node_community[node] = chioce
            # print(old_community, '->', self.node_community)
            if old_community == self.node_community:
                break
            else:
                old_community = self.node_community
        if init_community==self.node_community:
            return None
        else:
            mask = []
            mask_dict = {}
            for i,cm in enumerate(self.cms):
                if not cm.empty():
                    mask_dict[i] = len(mask)
                    mask.append(i)
            self.cms = [self.cms[i] for i in mask]
            self.node_community = [mask_dict[k] for k in self.node_community]

            edges = self.aggresive_community()
            return self.G.newGraph(edges)

    def aggresive_community(self):
        from _collections import defaultdict

        edges=defaultdict(float)
        for nodei in self.nodes:
            ci=self.node_community[nodei]
            for nodej in self.G.getNodeNeibours(nodei):
                cj=self.node_community[nodej]
                if(nodei<=nodej):
                    w=self.G.getPairWeight(nodei,nodej)
                    edges[(ci,cj)]+=w
                    # print(ci,cj,self.G.getPairWeight(nodei,nodej),'node %d,node %d'%(nodei,nodej))

        return edges


def louvain(G):
    clusters=[]
    while True:
        model=Louvain(G)
        G=model.run_one_level()
        if G is None:
            break
        else:
            clusters.append(model)
    return clusters
if __name__ == '__main__':
    from utils import makeChengfGraph
    G, Id2Index, Index2Id, _ = makeChengfGraph('data/15608544362450.json')

    Gm=GraphWrapper(G)
    clusters=louvain(Gm)
    print(clusters)