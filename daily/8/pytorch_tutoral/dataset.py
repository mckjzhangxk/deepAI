import torchtext.datasets as datasets
import torchtext.data as data
import torchtext.data.iterator as iterator

import spacy

spacy_de=spacy.load('de')
spacy_en=spacy.load('en')

SOS,EOS,UNK='<s>','</s>','<unk>'
def token_de(x):
    return [tok.text for tok in spacy_de.tokenizer(x)]
def token_en(x):
    return [tok.text for tok in spacy_en.tokenizer(x)]
src=data.Field(tokenize=token_de,unk_token=UNK)
tgt=data.Field(tokenize=token_en,init_token=SOS,eos_token=EOS,unk_token=UNK)
ds=datasets.TranslationDataset('.data/iwslt/de-en/IWSLT16.TEDX.tst2014.de-en',('.de','.en'),fields=(src,tgt))
src.build_vocab(ds.src,max_size=10000)
tgt.build_vocab(ds.trg,max_size=10000)

it=iterator.Iterator(ds,128,sort_key=lambda x: (len(x.src),len(x.trg)) )
if __name__=='__main__':
#    print(len(ds))
#    print(len(src.vocab))
#    print(len(tgt.vocab))
#    words=[tgt.vocab.itos[u] for u in range(len(tgt.vocab))]
#    print(words)
    for ii in it:
        print(ii.src.size()
        break
