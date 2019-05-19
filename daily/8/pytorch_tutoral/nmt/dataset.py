import numpy as np
import torch
from utils import makeMask,translation
import torchtext.data as data
import torchtext.datasets as datasets


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
    def __init__(self,path,exts=('.de','.en'),UNK='<unk>',SOS='<s>',EOS='</s>',TMAX=100,MIN_FREQ=2):
        '''

        '''
        import spacy

        spacy_x=spacy.load(exts[0][1:])
        spacy_y=spacy.load(exts[1][1:])


        def split_x(text):
            return [tok.text for tok in spacy_x.tokenizer(text)]
        def split_y(text):
            return [tok.text for tok in spacy_y.tokenizer(text)]

        SRC = data.Field(tokenize=split_x,unk_token=UNK,pad_token=UNK)
        TGT = data.Field(tokenize=split_y,init_token=SOS,eos_token=EOS,unk_token=UNK,pad_token=UNK)
        
        self.ds=datasets.TranslationDataset(
            path=path, 
            exts=exts,
            fields=(SRC, TGT),
            filter_pred=lambda x:len(x.src)<TMAX and len(x.trg)<TMAX)
        SRC.build_vocab(self.ds.src,min_freq=MIN_FREQ)
        TGT.build_vocab(self.ds.trg,min_freq=MIN_FREQ)
        self.SRC=SRC
        self.TGT=TGT
        self.padding_idx=self.SRC.vocab.stoi[UNK]
        self.sos=SOS
        self.unk=UNK
        self.eos=EOS
        
        self.sos_idx=self.TGT.vocab.stoi[SOS]
        self.eos_idx=self.TGT.vocab.stoi[EOS]

        print('src_padding:',self.padding_idx)
        self.padding_idx=self.TGT.vocab.stoi[UNK]
        print('tgt_padding:',self.padding_idx)
    def srcV(self):
            return len(self.SRC.vocab.itos)
    def tgtV(self):
         return len(self.TGT.vocab.itos)
    def __len__(self):
        return len(self.ds)
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
    def __next__(self):
        for data in self.iters:
            yield Batch(data.src,data.trg,self.ds.padding_idx)
if __name__=='__main__':

    ds=MyDataSet('../.data/iwslt/de-en/IWSLT16.TED.dev2010.de-en')
    print('Examples:',len(ds))
    print('Vs:',ds.srcV())
    print('Vt:',ds.tgtV())
    loader=MyDataLoader(ds,12000,shuffle=False)
    for i in range(2):
        for d in loader:
            print(d.x.size(),d.yin.size(),d.yout.size(),d.xmask.size(),d.ymask.size(),d.ntoken)
