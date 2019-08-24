import json
import networkx as nx
import numpy as np

def abc(data,seed=0,width=100,height=100,offset=20):

    G=nx.Graph()

    nodes=data['nodes']
    for node in nodes:
        G.add_node(node['phone'])

    edges=data['edges']
    for edge in edges:
        G.add_edge(edge['source'],edge['target'])


    pos=nx.spring_layout(G,seed=seed)
    # pos=nx.get_node_attributes(G,'pos')
    pos_array=[]
    for _,d in pos.items():
        pos_array.append(d)
    pos_array=np.array(pos_array)
    xmin,ymin=pos_array.min(axis=0)
    xmax, ymax = pos_array.max(axis=0)

    for node in nodes:
        id=node['phone']
        x,y=pos[id]
        x,y=(x-xmin)/(xmax-xmin),(y-ymin)/(ymax-ymin)

        node['x']=int(x*width)+offset
        node['y']=int(y*height)+offset
    return data