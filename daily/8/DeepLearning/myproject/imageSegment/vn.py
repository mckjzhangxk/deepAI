import  numpy as np
import functools
def bn(x,mv_u,mv_s,beta=0.99):
    u=np.mean(x)
    s=np.std(x)
    out=(x-u)/s
    mv_u=beta*mv_u+(1-beta)*u
    mv_s=beta*mv_s+(1-beta)*s

    return out,mv_u,mv_s

x=[1,2]
print(bn(x,0,0))