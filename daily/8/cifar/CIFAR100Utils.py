from __future__ import print_function

from six.moves import cPickle as pickle
import numpy as np
import os
from scipy.misc import imread
import platform

def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = load_pickle(f)
    X = datadict['data']
    Y = datadict['fine_labels']
    X = X.reshape(X.shape[0], 3, 32, 32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    return X, Y

def load_CIFAR100(ROOT):
  """ load all of cifar """
  Xtr,Ytr= load_CIFAR_batch(os.path.join(ROOT, 'train'))
  Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test'))
  return Xtr, Ytr, Xte, Yte

def load_dataset(flaten=False,one_hot=True):
    def _make_one_hot(d,C=10):
        return (np.arange(C)==d[:,None]).astype(np.int32)
    X_train,Y_train,X_test,Y_test=load_CIFAR100('CIFAR100_DATA')
    X_train/=255
    X_test/=255
    if flaten:
        X_train=X_train.reshape((-1,32*32*3))
        X_test = X_test.reshape((-1,32*32*3))
    if one_hot:
        Y_train=_make_one_hot(Y_train,C=100)
        Y_test = _make_one_hot(Y_test,C=100)
    print('\n-------------------------------------------------------------------------')
    print('load %d train Example,%d Test Example'%(X_train.shape[0],X_test.shape[0]))
    print('Train Images  Shape:'+str(X_train.shape))
    print('Train Labels  Shape:' + str(Y_train.shape))
    print('Test  Images  Shape:'+str(X_test.shape))
    print('Test  Labels  Shape:' + str(Y_test.shape))
    print('-------------------------------------------------------------------------')
    classes=list(range(100))
    return X_train,Y_train,X_test,Y_test,classes

# X_train,X_test,Y_train,Y_test,classes=load_dataset(flaten=False,one_hot=False)
# import matplotlib.pyplot as pyplot
# classidx=1
# idx=Y_train==classidx
# X=X_train[idx]
#
# for i in range(10):
#     pyplot.imshow(X[i])
#     pyplot.show()