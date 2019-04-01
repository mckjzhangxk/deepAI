import numpy as np
import numpy.random as npr

cc=npr.randint(0,255,(33,44,3,22))

for gt in range(100):
    # gt=i
    i=npr.randint(0,cc.shape[0],gt,np.int32)
    j=npr.randint(0,cc.shape[1],gt,np.int32)
    a=npr.randint(0,cc.shape[2],gt,np.int32)


    d=cc[i,j,a]
    print(d.shape)
    assert d.shape==(gt,cc.shape[3])