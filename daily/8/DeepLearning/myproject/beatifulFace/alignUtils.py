__author__ = 'mathai'
import numpy as np
import cv2


def distance(x, y):
    '''

    :param x:(m,f)
    :param y: (N,f)
    :return:z=(m,n)  z[i][j]=distance(x[i],y[j)
    '''

    a = x[:, None, :]
    b = y[None, :, :]

    return np.sqrt(np.square(a - b).sum(-1))


def kernel(a, b):
    '''

    :param a:(m,f)
    :param b:(N,f)
    :return::z=(m,n) z[i][j]=rdf(dist(a[i],b[j]))
    '''
    r = distance(a, b)

    return (r ** 2) * np.log(r + 1e-6)


def trainMatrix(X):
    '''
    构造用于训练的矩阵A
    :param X:(n,2)
    :return:(n+3,n+3)都
    '''

    n = X.shape[0]

    K = kernel(X, X)
    P = np.ones((n, 3), dtype=np.float32)
    P[:, 0:2] = X

    A = np.zeros((n + 3, n + 3), dtype=np.float32)
    A[:n, :n] = K
    A[:n, -3:] = P
    A[-3:, :n] = P.T

    return A


def uniform_grid(W, H,mask=None):
    X,Y= np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H), indexing='ij')
    Z1=None
    if mask is not None:
        xmin,xmax,ymin,ymax=mask
        X1=X[xmin:xmax,ymin:ymax].ravel()
        Y1=Y[xmin:xmax,ymin:ymax].ravel()
        Z1=np.stack((X1,Y1),axis=1)
    X = X.ravel()
    Y = Y.ravel()
    Z = np.stack((X,Y), axis=1)
    return Z,Z1


def fit(X, b):
    '''
    学一个从X 到b 的转换
    :param X: (n,2)
    :param b: (n,k)
    :return:(n+3,k)
    '''
    A = trainMatrix(X)

    o = np.zeros((3, b.shape[1]), dtype=np.float32)

    b = np.vstack((b, o))

    theta = np.linalg.solve(A, b)

    return theta


def predict(ctrl, theta, x):
    '''
    用系统转换函数（theta，ctrl） 把x  进行转换

    :param ctrl:(T,2) 个控制点，这些控制点用于训练theta
    :param theta:(T+3,2) T对应3上面控制点的数量,3分别是a1,a2,b,2表示生成x,y两个函数
    :param x:(N,2)
    :return:(N,2)
    '''

    T = theta.shape[0] - 3

    U = kernel(x, ctrl)

    return U.dot(theta[:T]) + x.dot(theta[T:T + 2]) + theta[T + 2][None, :]

from datetime import  datetime
def tps_wrap(img, pts, ctrl_pts,mask=None):
    '''
    可以这样理解，我想把img上的pt1 移到ctrl_pts，其他点如何生成
    :param img: 原图
    :param pts: (x,y) 正则化了
    :param ctrl_pts:(x,y) 正则化了
    :param dsize: 生成的图片的大小
    :return:
    '''

    H, W = img.shape[:2]

    theta = fit(ctrl_pts, pts)

    # (dsize.W*dsize.W,2)
    target_grid ,region_grid= uniform_grid(W,H,mask)
    
    # (dsize.W*dsize.W,2)
    if mask is None:
        src_pts = predict(ctrl_pts, theta, target_grid)
        src_pts.shape = (W,H, 2)

        mx = (src_pts[:, :, 0] * W).astype(np.float32).T
        my = (src_pts[:, :, 1] * H).astype(np.float32).T

    else:
        xmin,xmax,ymin,ymax=mask
        src_pts=predict(ctrl_pts,theta,region_grid)
        src_pts.shape = (xmax-xmin,ymax-ymin, 2)

        target_grid.shape=(W,H, 2)
        mx=(target_grid[:,:,0]* W).astype(np.float32)
        my=(target_grid[:,:,1]* H).astype(np.float32)

        mx[xmin:xmax,ymin:ymax]=src_pts[:,:,0]*W
        my[xmin:xmax,ymin:ymax]=src_pts[:,:,1]*H

        mx=mx.T
        my=my.T
    return cv2.remap(img,mx,my, cv2.INTER_CUBIC)





if __name__ == '__main__':
    N = 24
    x = np.random.rand(N, 2)
    y = np.random.rand(N, 2)
    b = np.random.rand(N, 2)

    assert distance(x, y).shape == (x.shape[0], y.shape[0])

    assert len(np.nonzero(distance(x, y) >= 0)[0]) == (x.shape[0] * y.shape[0])

    assert kernel(x, y).shape == (x.shape[0], y.shape[0])

    A = trainMatrix(x)
    assert (A.shape == (x.shape[0] + 3, x.shape[0] + 3))

    assert np.allclose(A, A.T)

    theta = fit(x, b)

    assert theta.shape == (x.shape[0] + 3, x.shape[1])

    for k in range(theta.shape[1]):
        t = theta[:, k]
        w = t[:x.shape[0]]

        assert np.allclose(np.abs(w.sum()), 0)

    b1 = predict(x, theta, x)

    print('误差', np.abs((b - b1)).max())

    X = uniform_grid(22, 33)

    assert X.shape == (22 * 33, 2)

    img = np.zeros((1024, 768, 3), np.uint8)
    tps_wrap(img, x, y, (1024, 768))