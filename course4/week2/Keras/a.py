import matplotlib.pyplot as plt

from scipy.misc import imread

img1=imread('images/happy-house.jpg')
plt.subplot(2,2,1)
plt.imshow(img1)
plt.title('图片1')
# plt.savefig('a.jpg')


plt.subplot(2,2,2)
img1=imread('images/house-members.png')
plt.title('图片1')
plt.imshow(img1)
# plt.savefig('a.jpg')


plt.subplot(2,2,3)
plt.title('图片1')
# plt.savefig('a.jpg')


plt.subplot(2,2,4)
plt.title('图片1')
plt.savefig('a.png')
def outputPicAndAcc(X,Y,Y_pred,filename,gridSize=(9,9)):
    m=X.shape[0]
    rows,cols=gridSize
    numExample=rows*cols


    for i in range(0,m,numExample):
        _X=X[i:i+numExample]
        _Y = Y[i:i + numExample]
        _Yhat=Y_pred[i:i + numExample]
        for row in range(1,rows+1):
            for col in range(1,cols+1):
                picIndex=(row-1)*cols+col-1
                plt.subplot(rows,cols,(row-1)*cols+col)
                plt.axis('off')

                if(picIndex<_X.shape[0]):
                    plt.imshow(_X[picIndex])
                    y,yhat=_Y[picIndex],_Yhat[picIndex]
                    plt.title('True label:'+str(y)+',with prob'+str(y*(yhat)+(1-y)*(1-yhat)))

    plt.savefig(filename)
