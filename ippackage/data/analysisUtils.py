import numpy as np


def pca(a):
    shape_a=a.shape
    a=a.reshape(-1,shape_a[-1])

    U, S, VT = np.linalg.svd(a, False)
    pri_axis=VT[0]*np.sign(VT[0][0])
    print(pri_axis,np.linalg.norm(pri_axis))
    a=a.dot(pri_axis)
    a=a.reshape(*(shape_a[:-1]))
    return a
def standardizeData(d,mode=None):
    def _standardize1(d):
        m = np.max(d, axis=0, keepdims=True) + 1e-14
        return d / m

    def _standardize2(d):
        u = np.mean(d, axis=0, keepdims=True)
        s = np.std(d, axis=0, keepdims=True) + 1e-14
        return (d - u) / s

    if mode=='STD':
        return _standardize2(d)
    if mode=='MAX':
        return _standardize1(d)
    return d
