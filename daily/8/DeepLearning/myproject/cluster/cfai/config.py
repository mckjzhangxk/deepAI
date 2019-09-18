import json,os
from collections import OrderedDict
def defaultConfig():
    d=OrderedDict()
    d['debug']=False
    basepath='/home/zhangxk/projects/deepAI/daily/8/DeepLearning/myproject/cluster/cfai/test'
    # basepath='/cfai/cluster'
    d['source']=os.path.join(basepath,'src/*.json')
    d['target']=os.path.join(basepath,'tgt')
    d['logfile']=os.path.join(basepath,'log')
    if not os.path.exists(d['logfile']):
        os.mkdir(d['logfile'])



    d['colors']=["#E25D68","#8BC34A","#03A9F4","#AB47BC","#ff9800","#3f51b5", "#EC407A",
                 "#009688", "#8D6E63",  "#FF7043","#FFCA28", "#4CAF50",
                 "#5C6BCD", "#827717", "#00695C", "#7E57C2", "#015798", "#FFEB3B",
                 "#89BEB2","#823935","#C9BA83","#DED38C","#DE9C53",
                 "#B2C8BB","#75794A","#458994","#725334","#F9CDAD"]
    d['refresh_interval']=1


    d['proposal_eps']=0.5
    d['pagerank_eps']=1e-2
    d['pagerank_alpha']=0.85
    d['pagerank_topk']=-1
    d['pagerank_maxiters']=300

    d['canvas_pos']=True
    d['canvas_seed'] = 0
    d['canvas_iters']=100

    d['eig_maxtry']=5
    d['eig_supportDim']=500


    d['max_clusters']=len(d['colors'])
    d['minNode']=30
    d['max_complex']=1.8

    # with open('../config.json','w') as fs:
    #     json.dump(d,fs,indent=1)
    return d
if __name__ == '__main__':
    defaultConfig()