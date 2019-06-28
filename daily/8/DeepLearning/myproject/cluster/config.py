import json
from collections import OrderedDict
def defaultConfig():
    d=OrderedDict()

    d['source']='test/src/*.json'
    d['target']='test/tgt'
    d['colors']=["#E25D68","#8BC34A","#03A9F4","#AB47BC","#ff9800","#3f51b5", "#EC407A", "#009688", "#8D6E63",  "#FF7043","#FFCA28", "#4CAF50", "#5C6BCD", "#827717", "#00695C", "#7E57C2", "#015798", "#FFEB3B"]
    d['refresh_interval']=1


    d['proposal_eps']=0.5
    d['pagerank_eps']=1e-2
    d['pagerank_alpha']=0.85
    d['pagerank_topk']=-1
    d['pagerank_matiters']=300
    with open('config.json','w') as fs:
        json.dump(d,fs,indent=1)
if __name__ == '__main__':
    defaultConfig()