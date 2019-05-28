import pickle
import argparse
import torch
from train import restore 
from model import makeModel,greedyDecoder
from utils import standardMask
import numpy as np
from metrics.evaluation_utils import blue


def loadVocab(path):
    with open(path,'rb') as fs:
        v=pickle.load(fs)
    return v['src'],v['tgt']

def parse():
    parser=argparse.ArgumentParser()
    parser.add_argument('--vocab',type=str)
    parser.add_argument('--datafile',type=str)
    parser.add_argument('--model_path',type=str)

    return parser.parse_args()

def readFile(datafile,exts=('.de','.en')):
    with open(datafile+exts[0]) as fs:
        src=fs.readlines()
    with open(datafile+exts[1]) as fs:
        tgt=fs.readlines()
    assert len(src)==len(tgt)
    return src,tgt
def myEvaluate(model,src,tgt,vocab_src,vocab_tgt,exts=('.de','.en'),UNK='<unk>',SOS='<s>',EOS='</s>'):
    '''
    src:list of string
    tgt:list of string 
    vocab_src:vocabulary for source
    vocab_tgt:vocabulary for target
    '''
    import spacy


    spacy_x=spacy.load(exts[0][1:])
    pad_idx=vocab_src[UNK]
    sos_idx=vocab_tgt[SOS]

    ypred=[]
    for s in src:
        src_tokens=[tok.text for tok in spacy_x.tokenizer(s)]
        src_encode=[vocab_src[tok] for tok in src_tokens]
        X=torch.from_numpy(np.array([src_encode]))
        Y=greedyDecoder(X,standardMask(X,pad_idx),model,startidx=sos_idx,unk=pad_idx)[0]

        wordlist=[vocab_tgt[idx] for idx in Y]
        if EOS in wordlist:
            wordlist=wordlist[:wordlist.index(EOS)]
        l=' '.join(wordlist)
        ypred.append(l)
    score=blue(ypred,tgt)
    return score

if __name__=='__main__':
    vocab_path='vocab'

    parser=parse()
    src_vocab,tgt_vocab=loadVocab(parser.vocab)
    src,tgt=readFile(parser.datafile)

    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model=makeModel(len(src_vocab),len(tgt_vocab))
    model=model.to(device)
    #restore(model,path=parser.model_path,tocpu=True)
    myscore=myEvaluate(model,src,tgt,src_vocab,tgt_vocab)

    print('score:',myscore)