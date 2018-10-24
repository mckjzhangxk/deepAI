import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats
import matplotlib as mtl

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# 画高斯分布
#mgrid 的用法,x~[-3,3] y~[-3,3]
X,Y=np.mgrid[-3:3:100j,-3:3:100j] #X,Y have shape 100x100,x is first axis

# xrange=np.linspace(-3,3,100)
# yrange=np.linspace(-6,6,100)
# X,Y=np.meshgrid(xrange,yrange)

Z=np.exp(-(X**2+Y**2)/2)/(2*np.pi)
fig = plt.figure(figsize=(5,5),facecolor='w')
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X,Y,Z,rstride=3,cstride=3,cmap=cm.Accent,linewidth=1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
