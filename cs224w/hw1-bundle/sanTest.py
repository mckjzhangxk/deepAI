import networkx as nx
from lavain import Community,GraphWrapper,Louvain,louvain
import numpy as np


def createGraph(ringNodes=4):
    G = nx.Graph()

    for i in range(ringNodes):
        s = 4 * i
        G.add_edge(s, s + 1, weight=1)
        G.add_edge(s, s + 2, weight=1)
        G.add_edge(s, s + 3, weight=1)
        G.add_edge(s + 1, s + 2, weight=1)
        G.add_edge(s + 1, s + 3, weight=1)
        G.add_edge(s + 2, s + 3, weight=1)

    for i in range(ringNodes):
        G.add_edge(3+i*4, (3+i*4+2)%(4*ringNodes), weight=1)
    return G

def sanity_test1():
    G=nx.Graph()
    G.add_edge(0, 0, weight=7)
    G.add_edge(0, 1, weight=4)
    G.add_edge(0, 2, weight=1)
    G.add_edge(0, 3, weight=1)

    G.add_edge(1, 1, weight=2)
    G.add_edge(1, 2, weight=1)

    G.add_edge(2, 2, weight=1)
    G.add_edge(2, 3, weight=3)

    G.add_edge(3, 3, weight=8)

    Gm=GraphWrapper(G,'weight')

    assert Gm.getNodeDegree(0)==20
    assert Gm.getNodeDegree(1)==9
    assert Gm.getNodeDegree(2)==7
    assert Gm.getNodeDegree(3)==20
    assert Gm.getTotalWeight()==56
    assert Gm.getPairWeight(0,0)==7
    assert Gm.getPairWeight(0,2) == 1

    assert Gm.getNodeNeibours(2)=={0:1,1:1,2:1,3:3}
    assert Gm.getNodeNeibours(1)=={0:4,1:2,2:1}


    assert Gm.getNodeWeightBetweenSet(0,[0,1,2,3])==26
    assert Gm.getNodeWeightBetweenSet(0, [1, 3]) == 10
    assert Gm.getNodeWeightBetweenSet(0, [2]) == 2
    assert Gm.getNodeWeightBetweenSet(3, [3,2,1]) == 22

def sanity_test2():
    G=nx.Graph()
    G.add_edge(0, 0, weight=7)
    G.add_edge(0, 1, weight=4)
    G.add_edge(0, 2, weight=1)
    G.add_edge(0, 3, weight=1)

    G.add_edge(1, 1, weight=2)
    G.add_edge(1, 2, weight=1)

    G.add_edge(2, 2, weight=1)
    G.add_edge(2, 3, weight=3)

    G.add_edge(3, 3, weight=8)

    Gm=GraphWrapper(G,'weight')

    c0 = Community(Gm, 0)
    c1 = Community(Gm, 1)
    c2 = Community(Gm, 2)
    c3 =Community(Gm, 3)


    assert c0.sigma_tot == 20
    assert c0.sigma_in == 14

    assert c1.sigma_tot==9
    assert c1.sigma_in==4

    assert c2.sigma_tot==7
    assert c2.sigma_in==2

    assert c3.sigma_tot==20
    assert c3.sigma_in==16



    a,b=c1.modularity_add(3)
    assert b==16

    a,b=c2.modularity_add(1)
    assert b==6

    a,b=c0.modularity_add(3)
    assert b==18

    a,b=c3.modularity_add(0)
    assert b==16

    a,b=c0.modularity_remove(0)
    assert b==14

    a, b = c1.modularity_remove(1)
    assert b == 4
    assert a<0
    a, b = c2.modularity_remove(2)
    assert b == 2
    assert a < 0
    a, b = c3.modularity_remove(3)
    assert b == 16
    assert a < 0

    ki=Gm.getNodeDegree(0)
    _, k_i_in =c2.modularity_add(0)
    _, k_i_out=c0.modularity_remove(0)
    c2.merge(c0,0,ki,k_i_in,k_i_out)
    assert c2.members==[2,0]
    assert c0.members==[]
    assert c2.sigma_tot==27
    assert c0.sigma_tot==0
    assert c2.sigma_in==18
    assert c0.sigma_in==0

    ki=Gm.getNodeDegree(1)
    _, k_i_in =c2.modularity_add(1)
    _, k_i_out=c1.modularity_remove(1)
    c2.merge(c1,1,ki,k_i_in,k_i_out)
    assert c2.members==[2,0,1]
    assert c1.members==[]
    assert c2.sigma_tot==36
    assert c1.sigma_tot==0
    assert c2.sigma_in==32
    assert c1.sigma_in==0

    ki=Gm.getNodeDegree(2)
    _, k_i_in =c3.modularity_add(2)
    _, k_i_out=c2.modularity_remove(2)
    c3.merge(c2,2,ki,k_i_in,k_i_out)
    assert c3.members==[3,2]
    assert c2.members==[0,1]
    assert c3.sigma_tot==27
    assert c2.sigma_tot==29
    assert c3.sigma_in==24
    assert c2.sigma_in==26

    ki=Gm.getNodeDegree(0)
    _, k_i_in = c3.modularity_add(0)
    _, k_i_out = c2.modularity_remove(0)
    c3.merge(c2,0,ki,k_i_in,k_i_out)
    assert c3.members==[3,2,0]
    assert c2.members==[1]
    assert c3.sigma_tot==47
    assert c2.sigma_tot==9
    assert c3.sigma_in==42
    assert c2.sigma_in==4


    ki=Gm.getNodeDegree(1)
    _, k_i_in = c3.modularity_add(1)
    _, k_i_out = c2.modularity_remove(1)
    c3.merge(c2,1,ki,k_i_in,k_i_out)
    assert c3.members==[3,2,0,1]
    assert c2.members==[]
    assert c3.sigma_tot==56
    assert c2.sigma_tot==0
    assert c3.sigma_in==56
    assert c2.sigma_in==0

def sanity_test3():
    G=nx.Graph()
    G.add_edge(0, 0, weight=7)
    G.add_edge(0, 1, weight=4)
    G.add_edge(0, 2, weight=1)
    G.add_edge(0, 3, weight=1)

    G.add_edge(1, 1, weight=2)
    G.add_edge(1, 2, weight=1)

    G.add_edge(2, 2, weight=1)
    G.add_edge(2, 3, weight=3)

    G.add_edge(3, 3, weight=8)

    Gm=GraphWrapper(G,'weight')
    model=Louvain(Gm)
    Gm1=model.run_one_level()

    assert model.cms[0].sigma_tot==29
    assert model.cms[1].sigma_tot == 27
    assert model.cms[0].sigma_in==26
    assert model.cms[1].sigma_in == 24

    assert len(Gm1.getNodes())==2

    assert Gm1.getNodeDegree(0)==29
    assert Gm1.getNodeDegree(1) == 27
    assert Gm1.getPairWeight(0,0)==13
    assert Gm1.getPairWeight(1, 1) == 12
    assert Gm1.getPairWeight(0, 1) == 3

    clusters=louvain(Gm)

def sanity_test4():
    rings=32
    G=createGraph(rings)
    G=GraphWrapper(G)
    clusters,graphs=louvain(G)

    assert len(clusters[0].cms)==rings
    G1=graphs[1]
    assert len(G1.getNodes())==rings

    assert G1.getTotalWeight()==rings*14
    for n in G1.getNodes():
        assert G1.getNodeDegree(n)==14
        assert G1.getPairWeight(n,n)==6
    for n in G1.getNodes():
        nb=G1.getNodeNeibours(n)
        assert len(nb)==3
        assert nb[n]==6
        assert nb[(n+1)%rings] == 1
        assert nb[(n-1)%rings] == 1

    G2=graphs[2]
    assert len(G2.getNodes())==rings//2
    for n in G2.getNodes():
        assert G2.getNodeDegree(n)==28
        assert G2.getPairWeight(n,n)==13

    print(clusters[1].get_community_members())
sanity_test1()
sanity_test2()
sanity_test3()
sanity_test4()
