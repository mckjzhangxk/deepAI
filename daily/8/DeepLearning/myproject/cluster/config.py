import json
from collections import OrderedDict
def defaultConfig():
    d=OrderedDict()
    d['debug']=True
    d['source']='test/src/*.json'
    d['target']='test/tgt'
    d['colors']=["#E25D68","#8BC34A","#03A9F4","#AB47BC","#ff9800","#3f51b5", "#EC407A", "#009688", "#8D6E63",  "#FF7043","#FFCA28", "#4CAF50", "#5C6BCD", "#827717", "#00695C", "#7E57C2", "#015798", "#FFEB3B"]
    d['refresh_interval']=1


    d['proposal_eps']=0.5
    d['pagerank_eps']=1e-2
    d['pagerank_alpha']=0.85
    d['pagerank_topk']=-1
    d['pagerank_maxiters']=300

    d['canvas_pos']=False
    d['canvas_width']=1024
    d['canvas_height']=768
    d['canvas_seed'] = 0
    d['canvas_pad']=0

    d['eig_maxtry']=5
    d['eig_supportDim']=500


    d['max_clusters']=len(d['colors'])
    d['minNode']=10
    d['max_complex']=1.8

    with open('config.json','w') as fs:
        json.dump(d,fs,indent=1)
if __name__ == '__main__':
    defaultConfig()