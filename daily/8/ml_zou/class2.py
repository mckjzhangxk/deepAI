import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats
import matplotlib as mpl
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def drawNormDistribution(X,Y,Z,msg):
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    ax.plot_surface(X,Y,Z,rstride=2,cstride=2,cmap=cm.Accent)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.title(msg)
    fig.show()
X,Y=np.mgrid[-6:6:100j,-6:6:100j]

#标准正太分布
Z=(0.5/np.pi)*np.exp(-0.5*(X**2+Y**2))
drawNormDistribution(X,Y,Z,r'$ \mu=0,\ \sigma=I$')

# 正太分布,u=1,2
Z=(0.5/np.pi)*np.exp(-0.5*((X-1)**2+(Y-2)**2))
drawNormDistribution(X,Y,Z,r'$\mu=[1,2],\ \sigma=I$')

# cov=[3 0,0 1]
Z=(0.5/np.pi/np.sqrt(3))*np.exp(-0.5*((X**2)/3+Y**2))
drawNormDistribution(X,Y,Z,r'$ \mu=0,\ \sigma=[3,0,0,1]$')


# cov=[1 0.5,0.5 1]
cov=np.array([[1,.5],[.5,1]])
inv_cov=np.linalg.inv(cov)
Z=(0.5/np.pi/np.sqrt(0.75))*np.exp(-0.5*(inv_cov[0,0]*X**2+inv_cov[1,1]*Y**2+2*inv_cov[0,1]*X*Y))
drawNormDistribution(X,Y,Z,r'$ \mu=0,\ \sigma=[1,.5,.5,1]$')

