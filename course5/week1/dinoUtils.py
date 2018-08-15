import numpy as np

def loadDatasets():
    EOF='<EOF>'
    with open('Dino/dinos.txt') as f:
        chars=f.read() #read all file content
    vocabs=set(chars)
    char2idx={ch:i for i,ch in enumerate(sorted(vocabs))}
    char2idx[EOF]=len(char2idx)
    idx2char={i:ch for i,ch in enumerate(sorted(vocabs))}
    idx2char[len(idx2char)]=EOF
    vocabs.add(EOF)


    Tmax=-1
    lines=chars.split('\n')
    for l in lines:
        length=len(l)
        if length>Tmax:Tmax=length



    #after for-loop coupus have len=# of examples
    coupus=[]
    for l in lines:
        length = len(l)
        sentence=[char2idx[ch] for ch in l]+[char2idx[EOF]]*(Tmax-length+1)
        assert len(sentence)==Tmax+1
        coupus.append(sentence)

    X,Y=[],[]

    for s in coupus:
        X.append(s[0:-1])
        Y.append(s[1:])

    #===============one_hot representation begin=======================
    X_train=np.expand_dims(np.array(X),axis=2)
    Y_train=np.expand_dims(np.array(Y),axis=2)
    c=np.arange(len(vocabs))
    X_train=(X_train==c).astype(np.int32)
    Y_train=(Y_train==c).astype(np.int32)
    #===============one_hot representation end =======================

    return X_train,Y_train,char2idx,idx2char,vocabs
def showSentence(X,idx2char):
    T=X.shape[0]
    _s=''
    for t in range(T):
        _s=_s+idx2char[np.argmax(X[t])]
    return _s
X_train,Y_train,char2idx,idx2char,vocabs=loadDatasets()
# idx=99
# print(showSentence(X_train[idx],idx2char))
# print(showSentence(Y_train[idx],idx2char))
