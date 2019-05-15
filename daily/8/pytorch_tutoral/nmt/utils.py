import torch
def subseqenceMask(x):
    '''
    x:(N,T)
    return:
        (1,T,T)
    '''
    T=x.size(-1)
    return torch.tril(torch.ones((T,T))).unsqueeze(0).byte()
def standardMask(x,paddingidx):
    '''
    x:(N,T)
    paddingidx:set coresponding mask=0 when occur
            paddingidx
    return:(N,1,T)
    '''
    return (x!=paddingidx).unsqueeze(1).byte()
def makeMask(x,y,paddingidx):
    xmask=standardMask(x,paddingidx)
    ymask=standardMask(y,paddingidx)&subseqenceMask(y)
    return xmask,ymask

if __name__=='__main__':
    import matplotlib.pyplot as plt

    N,T=38,25

    X=torch.randint(0,10,(N,T))

    mask1=standardMask(X,0)
    mask2=subseqenceMask(X)
    assert mask1.shape==(N,1,T)
    assert mask2.shape==(1,T,T)


    plt.figure()
    plt.imshow(mask1[0],cmap='gray')
    plt.figure()
    plt.imshow(mask2[0],cmap='gray')
    plt.figure()
    plt.imshow(mask1[0]&mask2[0],cmap='gray')

    plt.show()
