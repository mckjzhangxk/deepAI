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
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    return X, Y

def load_CIFAR10(ROOT):
  """ load all of cifar """
  xs = []
  ys = []
  for b in range(1,6):
    f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
    X, Y = load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  del X, Y
  Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
  return Xtr, Ytr, Xte, Yte

def load_dataset(flaten=False,one_hot=True,filename='../../AI_database/cifar/CIFAR10_DATA'):
    def _make_one_hot(d,C=10):
        return (np.arange(C)==d[:,None]).astype(np.int32)
    X_train,Y_train,X_test,Y_test=load_CIFAR10(filename)
    X_train/=255
    X_test/=255
    if flaten:
        X_train=X_train.reshape((-1,32*32*3))
        X_test = X_test.reshape((-1,32*32*3))
    if one_hot:
        Y_train=_make_one_hot(Y_train)
        Y_test = _make_one_hot(Y_test)
    print('\n-------------------------------------------------------------------------')
    print('load %d train Example,%d Test Example'%(X_train.shape[0],X_test.shape[0]))
    print('Train Images  Shape:'+str(X_train.shape))
    print('Train Labels  Shape:' + str(Y_train.shape))
    print('Test  Images  Shape:'+str(X_test.shape))
    print('Test  Labels  Shape:' + str(Y_test.shape))
    print('-------------------------------------------------------------------------')
    classes=[]
    return X_train,Y_train,X_test,Y_test,classes
# load_dataset(flaten=False)
if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test, classes=load_dataset(flaten=True,one_hot=False,filename='/home/zxk/AI/data/cifar/CIFAR10_DATA')