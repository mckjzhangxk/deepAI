import numpy as np

def _initial_parameters(n_x,n_a,n_y):
    params={}
    params['Waa'] = np.random.randn(n_a,n_a)
    params['Wax'] = np.random.randn(n_a,n_x)
    params['ba'] = np.random.randn(n_a,1)
    params['Wya'] = np.random.randn(n_y,n_a)
    params['by'] = np.random.randn(n_y,1)
    return params
def _initial_gradients(params):
    grads={'d'+key:np.zeros_like(value) for key,value in params.items()}
    grads['da_next']=0
    return grads
def softmax(x):
    '''
        x have shape [n,m]
    :param x:
    :return:
    '''
    x=x-np.max(x,axis=0,keepdims=True)
    expx=np.exp(x)
    expsum=np.sum(expx,axis=0,keepdims=True)
    return expx/expsum
def _unpack(parameter):
    return parameter['Waa'],parameter['Wax'],parameter['Wya'],parameter['ba'],parameter['by']
def _rnn_step_forward(xt,a_prev,parameter):
    '''


    :param xt:shape [nx,m]
    :param a_prev: [na_m]
    :param parameter: Waa,Wax,Way,bx,by
    :return: cache all input and parameter,and a(i+1)
    and a y_predition
    '''

    Waa, Wax, Wya, ba, by=_unpack(parameter)

    a_out=np.tanh(Waa.dot(a_prev)+Wax.dot(xt)+ba)
    ypred=softmax(Wya.dot(a_out)+by)
    cache=xt,a_prev,a_out,parameter
    return a_out,ypred,cache
def _rnn_step_backward(dy,cache,gradients):
    '''


    :param dy:shape[n_y,m]
    :param cache:xt,a_prev,parameter
    :param gradients: dWaa,dWy,dWax,dba,dby,da_next
    :return:gradients
    '''
    xt,a_prev,a_out,parameter=cache
    Waa, Wax, Wya, ba, by = _unpack(parameter)

    #from linear prediction
    dWya=dy.dot(a_out.T)
    dby=np.sum(dy,axis=1,keepdims=True)
    da_next=Wya.T.dot(dy)+gradients['da_next']

    #from rnn units
    dz=da_next*(1-a_out**2)
    dWaa=dz.dot(a_prev.T)
    dWax=dz.dot(xt.T)
    dba=np.sum(dz,axis=1,keepdims=True)

    gradients['da_next']=Waa.T.dot(dz)
    gradients['dWaa']+=dWaa
    gradients['dWax'] += dWax
    gradients['dba'] += dba
    gradients['dWya'] += dWya
    gradients['dby'] += dby

    return gradients



def _rnn_forward(x,a_prev,parameter):
    '''


    :param x: shape [n_x,m,T]
    :param a_prev: shape [n_a,m]
    :param parameter: Waa,Wax,Way,ba,by
    :return: y_pred shape:[n_y,m,T],
        caches:a list of all cache
    '''

    n_x,m,T=x.shape
    n_y,n_a=parameter['Wya'].shape

    #the return value
    y_pred=np.zeros((n_y,m,T))
    a_out=np.zeros((n_a,m,T))
    caches=[]


    for t in range(T):
        a_prev,yhat,cache=_rnn_step_forward(x[:,:,t],a_prev,parameter)

        y_pred[:,:,t]=yhat
        a_out[:,:,t]=a_prev
        caches.append(cache)
    return y_pred,a_out,caches

def _rnn_backward(dy,caches,param):
    '''


    :param dy:shape[n_c,m,T]
    :param caches: cahces of rnn_forward
    :return: gradients
    '''
    n_y,m,T=dy.shape
    gradients=_initial_gradients(param)
    for t in reversed(range(T)):
        gradients=_rnn_step_backward(dy[:,:,t],caches[t],gradients)
    return gradients

def _computeLoss(yhat,y):
    '''

    :param yhat:
    :param y:[n_y,m,T]
    :return:
    '''
    #shape mxT
    prob_of_trueLabel=np.sum(yhat*y,axis=0)
    prob_of_trueLabel=prob_of_trueLabel.ravel()

    loss=np.mean(-np.log(prob_of_trueLabel))
    return loss
from keras.applications.vgg16 import  VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
# VGG16()
VGG19()
ResNet50()
InceptionV3()
m,n_x,n_a,n_y,T=100,27,32,10,60

x=np.random.randn(n_x,m,T)
a_prev=np.random.randn(n_a,m)
params=_initial_parameters(n_x,n_a,n_y)
y_pred,a_out,caches=_rnn_forward(x,a_prev,params)

dy=np.random.randn(n_y,m,T)
gradients=_rnn_backward(dy,caches,params)