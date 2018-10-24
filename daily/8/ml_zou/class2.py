import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats
import matplotlib as mpl
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def drawNormDistribution(fig,X,Y,Z,index):
    ax=fig.add_subplot(index,projection='3d')
    ax.plot_surface(X,Y,Z,rstride=2,cstride=2,cmap=cm.Accent)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
X,Y=np.mgrid[-6:6:100j,-6:6:100j]
fig=plt.figure()
#标准正太分布
Z=(0.5/np.pi)*np.exp(-0.5*(X**2+Y**2))
drawNormDistribution(fig,X,Y,Z,221)
plt.title(r'$ \mu=0,\ \sigma=I$')

# 正太分布,u=1,2
Z=(0.5/np.pi)*np.exp(-0.5*((X-1)**2+(Y-2)**2))
drawNormDistribution(fig,X,Y,Z,222)
plt.title(r'$\mu=[1,2],\ \sigma=I$')
# cov=[3 0,0 1]
Z=(0.5/np.pi/3)*np.exp(-((X**2)/3+Y**2))
drawNormDistribution(fig,X,Y,Z,223)
plt.title(r'$ \mu=0,\ \sigma=[3,0,0,1]$')

# cov=[1 0.5,0.5 1]
Z=(0.5/np.pi/0.75)*np.exp(-(X**2+Y**2+X*Y))
drawNormDistribution(fig,X,Y,Z,224)
plt.title(r'$ \mu=0,\ \sigma=[1,.5,.5,1]$')
fig.show()
