import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn import metrics

from tensorflow.examples.tutorials.mnist import input_data
def plotROC(fp,tp,auc,thresholds,ratio=0.02):
    plt.figure(facecolor='w')
    plt.plot(fp,tp,label='ROC Curve,Auc:%f'%(auc))
    bestIndex=np.argwhere(fp<=ratio)[-1][0]
    c_fp,c_tp=fp[bestIndex],tp[bestIndex]
    print('best rule is score<=%f,under this rule,TPR is %f,FPR is %f'%(1-thresholds[bestIndex],c_tp,c_fp))
    plt.plot([c_fp],[c_tp],'ro',label='chioce with fpr <=%.3f'%ratio)
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title('ROC analysis')
    plt.grid()
    plt.legend()


def ROC_AUC(y, y_score, flag='multi'):
    '''
    when Flag is multi,return micro ROC and ROC for every classes
    m_fp,m_tp,m_thresholds:1-D array,the fpr,tpr,threshold,auc 
    fp,tp,thresholds,auc:fp[i],tp[i] is for i'th classes


    '''
    if flag == 'multi':
        n_class = y_score.shape[1]
        fp, tp, thresholds, auc = {}, {}, {}, {}
        for i in range(n_class):
            fp[i], tp[i], thresholds[i] = metrics.roc_curve(y[:, i], y_score[:, i], drop_intermediate=False)
            auc[i] = metrics.auc(fp[i], tp[i])
            m_fp, m_tp, m_thresholds = metrics.roc_curve(y.ravel(), y_score.ravel())
            m_auc = metrics.auc(m_fp, m_tp)

            return (m_fp, m_tp, m_thresholds, m_auc, fp, tp, thresholds, auc)
    else:
        m_fp, m_tp, m_thresholds = metrics.roc_curve(y, y_score)
        m_auc = metrics.auc(m_fp, m_tp)
        return (m_fp, m_tp, m_thresholds, m_auc)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

#     print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
mnist_data_path='../AI_database/mnist/MNIST_DATA'
def imshow(X,Y=None,classes=None):
    '''
        show Batch of image in grids sqrt(h) x sqrt(w)
        X is a numpy array,size (m,h,w,c)
        Y is a numpy array,size (m,#classes)
    '''
    m=X.shape[0]
    gridSize=int(m**0.5)
    for i in range(0,gridSize):
        for j in range(0,gridSize):
            _idx=i*gridSize+j
            im=X[_idx]
            plt.subplot(gridSize,gridSize,_idx+1)
            plt.axis('off')
            plt.imshow(im)
            if Y is not None:
                label=classes[np.argmax(Y[_idx])]
                plt.title(label)

def load_dataset(flaten=False,one_hot=True):
    def _make_one_hot(d,C=10):
        return (np.arange(C)==d[:,None]).astype(np.int32)

    mnist=input_data.read_data_sets(mnist_data_path)
    X_train,Y_train=mnist.train.images,mnist.train.labels
    X_test,Y_test=mnist.test.images,mnist.test.labels

    if flaten==False:
        X_train=X_train.reshape((-1,28,28,1))
        X_test = X_test.reshape((-1, 28, 28,1))
    if one_hot:
        Y_train = _make_one_hot(Y_train)
        Y_test=_make_one_hot(Y_test)


    print('\n-------------------------------------------------------------------------')
    print('load %d train Example,%d Test Example'%(X_train.shape[0],X_test.shape[0]))
    print('Train Images  Shape:'+str(X_train.shape))
    print('Train Labels  Shape:' + str(Y_train.shape))
    print('Test  Images  Shape:'+str(X_test.shape))
    print('Test  Labels  Shape:' + str(Y_test.shape))
    print('-------------------------------------------------------------------------')
    return (X_train,Y_train,X_test,Y_test)