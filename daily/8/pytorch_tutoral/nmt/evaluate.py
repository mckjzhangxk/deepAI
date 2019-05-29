import pickle
import argparse
import torch
from train import restore 
from model import makeModel,greedyDecoder
from utils import standardMask
import numpy as np
from metrics.evaluation_utils import blue
from tqdm import tqdm

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

def readFile(datafile,exts=('.de','.en'),maxLen=100):
    with open(datafile+exts[0]) as fs:
        src=fs.readlines()
    with open(datafile+exts[1]) as fs:
        tgt=fs.readlines()
    import spacy
    spacy_x=spacy.load(exts[0][1:])
    spacy_y = spacy.load(exts[1][1:])

    retsrc,rettgt=[],[]
    for s,t in zip(src,tgt):
        _s=[tok.text for tok in spacy_x.tokenizer(s.strip())]
        _t=[tok.text for tok in spacy_y.tokenizer(t.strip())]
        if len(_s)>maxLen or len(_t)>maxLen:continue
        retsrc.append(_s)
        rettgt.append(_t)

    assert len(retsrc)==len(rettgt)
    return retsrc,rettgt

def myEvaluate(model,src,tgt,vocab_src,vocab_tgt,
               exts=('.de','.en'),
               UNK='<unk>',SOS='<s>',EOS='</s>',
               device='cpu',TMax=100):
    '''
    src:list of string
    tgt:list of string 
    vocab_src:vocabulary for source
    vocab_tgt:vocabulary for target
    '''


    model.eval()

    inv_vocab_tgt={v:k for k,v in vocab_tgt.items()}


    pad_idx=vocab_src[UNK]
    sos_idx=vocab_tgt[SOS]
    eos_idx=vocab_tgt[EOS]

    ypred=[]

    def batch(sentence_list,vocab,padidx,eos_idx=None):
        maxlen=max(len(x) for x in sentence_list)
        ret=[]
        for s in sentence_list:
            src_encode = [vocab[tok] for tok in s]
            if eos_idx is not None:src_encode.append(eos_idx)
            src_encode.extend([padidx]*(maxlen-len(src_encode)))
            ret.append(src_encode)
        return np.array(ret)
    def translateBack(decode_result,vocab,EOS):
        ret=[]

        for y in decode_result:
            wordlist = [vocab[idx] for idx in y]
            if EOS in wordlist:
                wordlist = wordlist[:wordlist.index(EOS)]
            l = ' '.join(wordlist)
            ret.append(l)
        return ret


    bs=16
    ytrue=[]
    for i in tqdm(range(0,len(src),bs)):
        X=batch(src[i:i+bs],vocab=vocab_src,padidx=pad_idx)
        X=torch.from_numpy(X).to(device)
        Y=greedyDecoder(X,standardMask(X,pad_idx),model,startidx=sos_idx,unk=pad_idx)
        ypred.extend(translateBack(Y,inv_vocab_tgt,EOS))
        ytrue.extend([' '.join(s) for s in tgt[i:i+bs]])

    score=blue([s.lower() for s in ypred],[s.lower() for s in ytrue])

    return score

if __name__=='__main__':
    # --model_path=models --datafile=../.data/iwslt/de-en/IWSLT16.TED.tst2014.de-en --vocab=vocab
    #29.15,30.8,24.98
    parser=parse()
    src_vocab,tgt_vocab=loadVocab(parser.vocab)
    src,tgt=readFile(parser.datafile)

    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model=makeModel(len(src_vocab),len(tgt_vocab))
    model=model.to(device)
    restore(model,path=parser.model_path,tocpu=False)
    myscore=myEvaluate(model,src,tgt,src_vocab,tgt_vocab,device=device)
    print('score:',myscore)