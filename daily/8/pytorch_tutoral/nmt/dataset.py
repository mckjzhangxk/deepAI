import numpy as np
import torch
from utils import makeMask,translation
import torchtext.data as data
import torchtext.datasets as datasets
import argparse


def dataGen(batch,T,V,unk=0):
    data=np.random.randint(0,V,(batch,T))

    x=torch.from_numpy(data)
    y=x.clone()
    for i in range(10000): 
        yield Batch(x,y,unk)
class Batch:
    def __init__(self,x,y,padding_idx):
        '''
        x:(N,T1)
        y:(N,T2)
        '''
        self.x=x
        self.yin=y[:,:-1]
        self.yout=y[:,1:]
        self.xmask,self.ymask=makeMask(self.x,self.yin,padding_idx)
        self.ntoken=torch.sum(y!=padding_idx)
class MyDataSet:
    def __init__(self,path,SRC,TGT,exts=('.de','.en'),UNK='<unk>',SOS='<s>',EOS='</s>',TMAX=100):
        '''

        '''
        self.ds=datasets.TranslationDataset(
            path=path, 
            exts=exts,
            fields=(SRC, TGT),
            filter_pred=lambda x:len(x.src)<TMAX and len(x.trg)<TMAX)
        
        self.src=self.ds.src
        self.tgt=self.ds.trg

        self.SRC=SRC
        self.TGT=TGT
        
        self.sos=SOS
        self.unk=UNK
        self.eos=EOS
        
    def build_special_word_idx(self):
        self.sos_idx=self.TGT.vocab.stoi[self.sos]
        self.eos_idx=self.TGT.vocab.stoi[self.eos]
        self.padding_idx=self.SRC.vocab.stoi[self.unk]
        print('src_padding:',self.padding_idx)
        self.padding_idx=self.TGT.vocab.stoi[self.unk]
        print('tgt_padding:',self.padding_idx)
    def srcV(self):
            return len(self.SRC.vocab.itos)
    def tgtV(self):
         return len(self.TGT.vocab.itos)
    def __len__(self):
        return len(self.ds)
    

    @staticmethod
    def getDataSet(trainfile,testfile,exts=('.de','.en'),UNK='<unk>',SOS='<s>',EOS='</s>',TMAX=100,MIN_FREQ=2):
        import spacy
        spacy_x=spacy.load(exts[0][1:])
        spacy_y=spacy.load(exts[1][1:])


        def split_x(text):
            return [tok.text for tok in spacy_x.tokenizer(text)]
        def split_y(text):
            return [tok.text for tok in spacy_y.tokenizer(text)]

        SRC = data.Field(tokenize=split_x,unk_token=UNK,pad_token=UNK)
        TGT = data.Field(tokenize=split_y,init_token=SOS,eos_token=EOS,unk_token=UNK,pad_token=UNK)
         
        train=MyDataSet(trainfile,SRC,TGT,exts=exts,UNK=UNK,SOS=SOS,EOS=EOS,TMAX=TMAX)
        test= MyDataSet(testfile,SRC,TGT,exts=exts,UNK=UNK,SOS=SOS,EOS=EOS,TMAX=TMAX)
        
        SRC.build_vocab(train.src,min_freq=MIN_FREQ)
        TGT.build_vocab(train.tgt,min_freq=MIN_FREQ)
        
        train.build_special_word_idx()
        test.build_special_word_idx()

        return train,test

class MyDataLoader:
    def __init__(self,ds,batch_size,shuffle=False,device='cpu'):
        self.maxSeqlen=0
        def batch_size_fn(new,count,validbatch):
                    if count==1:
                        self.maxSeqLen=0
                    self.maxSeqLen=max(self.maxSeqLen,len(new.src),len(new.trg))
                    return self.maxSeqLen*count

        self.iters=data.Iterator(ds.ds,
                            batch_size=batch_size, 
                            sort_key=lambda x:(len(x.src),len(x.trg)),
                            train=True, 
                            repeat=False, 
                            shuffle=shuffle,
                            batch_size_fn=batch_size_fn,
                            device=device
                           )
        self.batch_size=batch_size
        self.shuffle=shuffle 
        self.ds=ds
    def rebatch(self,data_iter):
        for d in data_iter:
            yield (d.src.transpose(0,1),d.trg.transpose(0,1))
    def translate_src(self,x):
        return translation(x,self.ds.SRC,self.ds.eos)
    def translate_tgt(self,y):
        return translation(y,self.ds.TGT,self.ds.eos)
    def __iter__(self):
        for data in self.rebatch(self.iters):
            yield Batch(*data,self.ds.padding_idx)
        return self

def createVocab(datafile,output,exts=('.de','.en'),UNK='<unk>',SOS='<s>',EOS='</s>',MIN_FREQ=2,TMAX=100):
    import spacy
    spacy_x=spacy.load(exts[0][1:])
    spacy_y=spacy.load(exts[1][1:])


    def split_x(text):
        return [tok.text for tok in spacy_x.tokenizer(text)]
    def split_y(text):
        return [tok.text for tok in spacy_y.tokenizer(text)]

    SRC = data.Field(tokenize=split_x,unk_token=UNK,pad_token=UNK)
    TGT = data.Field(tokenize=split_y,init_token=SOS,eos_token=EOS,unk_token=UNK,pad_token=UNK)
    ds=datasets.TranslationDataset(
        path=datafile, 
        exts=exts,
        fields=(SRC, TGT),
        filter_pred=lambda x:len(x.src)<TMAX and len(x.trg)<TMAX)
    SRC.build_vocab(ds.src,min_freq=MIN_FREQ)
    TGT.build_vocab(ds.trg,min_freq=MIN_FREQ)

    vocab_src=SRC.vocab.stoi
    vocab_tgt=TGT.vocab.stoi

    print('src have length',len(vocab_src))
    print('tgt have length',len(vocab_tgt))
    
    save_dict={'src':vocab_src,'tgt':vocab_tgt}

    import pickle
    with open(output,'wb') as fs:
        pickle.dump(save_dict,fs)

def parse():
    parser=argparse.ArgumentParser()
    parser.add_argument('--datafile','-d',type=str,help='datafile path')
    parser.add_argument('--minfreq',type=int)
    parser.add_argument('--output','-o',type=str)
    return parser.parse_args()



if __name__=='__main__':
    #python3 dataset.py -d ../.data/iwslt/de-en/train --minfreq=2 -o=vocab
    parser=parse()
    print(parser)

    createVocab(parser.datafile,MIN_FREQ=parser.minfreq,output=parser.output)
#    train,test=MyDataSet.getDataSet('../.data/iwslt/de-en/IWSLT16.TED.dev2010.de-en','../.data/iwslt/de-en/IWSLT16.TED.tst2012.de-en',TMAX=1000,MIN_FREQ=1)
#    print('Examples:',len(train))
#    print('Vs:',train.srcV())
#    print('Vt:',train.tgtV())
#
    

#    loader=MyDataLoader(train,400,shuffle=False)
#    for i in range(1):
#        for d in loader:
#            print(d.x.size(),d.yin.size(),d.yout.size(),d.xmask.size(),d.ymask.size(),d.ntoken)
#            print(loader.translate_tgt(d.yout))
#            break
