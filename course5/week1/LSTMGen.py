import tensorflow as tf
import numpy as np
from keras.layers import Dense,LSTM,Lambda,Input,RepeatVector
from keras.optimizers import Adam,rmsprop
from keras.losses import categorical_crossentropy
from keras.models import Model,load_model
import keras.backend as K
from dinoUtils import *

def one_hot_Input(y):
    '''
    y=[p1,p2,p3,....pC]
      --->[p_maxProbindex]
            ---->[0,0,0,...1...0]
    :param y:a tensor of shape[?,n_y]
    :return: a keras input layer with shape [?,1,n_y]
    '''
    n_y=y.shape[1]
    one_hot_vector=tf.one_hot(tf.argmax(y,axis=-1),n_y) #shape[?,n_y]
    o=RepeatVector(1)(one_hot_vector)
    return o

class Configure():
    def __init__(self):
        self.X_train, self.Y_train, self.char2idx, self.idx2char, self.vocabs = loadDatasets()

        #base hyper parameter for define model
        self.T=self.X_train.shape[1]
        self.vocabsize=self.X_train.shape[2]
        self.n_a=32
        self.cell=LSTM(self.n_a,return_state=True,name='lstm')
        self.densor=Dense(self.vocabsize,activation='softmax',name='out')


        #base training hyper parameter
        self.optimizer=rmsprop(lr=0.002,decay=0.001)
        self.batchsize=128
        self.epochs=200
        self.modelPath = 'outputs/dinos_predict.h5'


    def getModelConfig(self):
        return (self.T,self.vocabsize,self.n_a,self.cell,self.densor)
    def getTrainData(self):
        m=self.X_train.shape[0]
        a0=np.zeros((m,self.n_a))
        c0=a0.copy()

        inputs=[self.X_train,a0,c0]
        outputs=list(self.Y_train.copy().transpose([1,0,2]))

        return (inputs,outputs)
    def initialSample(self):
        m=self.X_train.shape[0]

        a0=np.zeros((m,self.n_a))
        c0=a0.copy()
        x=np.zeros((1,1,self.vocabsize))
        # x[0][0][self.char2idx['b']]=1

        inputs=[x,a0,c0]
        return inputs

def TrainGenModel(config:Configure,summary=False):
    '''
    define a model inputs is a [m,T,vocab] time sequence
    pass it though a LSTM,return T's predicts(y)
    :param config:
    :return: a trained LSTM model,
    for train this model,you should fetch
    Inputs:
        X:[m,T,vocabSize]
        a0:[m,n_a]
        c0:[m,n_a]
        Y:a list of len()=T,Y(i) have shape [m,vocabSize]
    '''
    T,vocabSize,n_a,cell,densor=config.getModelConfig()
    X=Input(shape=(T,vocabSize,),name='train_input')
    a0=Input(shape=(n_a,),name='a0')
    c0 = Input(shape=(n_a,),name='c0')

    #define the sequence
    a,c=a0,c0
    outputs=[]
    for t in range(T):
        x=Lambda(lambda x:x[:,t:t+1,:])(X)
        a,_,c=cell(x,initial_state=[a,c]) #a,c have shape (?,vocab)
        y=densor(a) #y have shape (?,vocab)

        outputs.append(y)
    model=Model(inputs=[X,a0,c0],outputs=outputs)
    if summary:
        model.summary()
    return model

def PredictModel(config:Configure,summary=False):
    '''
    define a sample model inputs is a [m,1,vocab]  sequence
    pass it though a LSTM,get a prediction y,then convert y
    to input for the next lstm forward step
    y(1)        y(2)
    ^           ^
    |           |
    LSTM--->   LSTM--->
    ^           ^
    |           |
    x         convert(y(1))
    :param config:
    :return: a trained LSTM model
    for using this model:fecth the data of this form
        Inputs:
        X:[m,1,vocabSize]
        a0:[m,n_a]
        c0:[m,n_a]
    '''
    T,vocabSize,n_a,cell,densor=config.getModelConfig()

    #all inputs to tensor
    X=Input(shape=(1,vocabSize,),name='predict_input')
    a0=Input(shape=(n_a,),name='predict_a0')
    c0 = Input(shape=(n_a,),name='predict_c0')


    #define the sequence
    a,c=a0,c0
    outputs=[]
    x=X  #x is the change every step
    for t in range(T):
        a,_,c=cell(x,initial_state=[a,c]) #a,c have shape (?,vocab)
        y=densor(a) #y have shape (?,vocab)
        outputs.append(y)

        #y is a normal tf's tensor, using Lambda to wrap it,can be using in keras!
        x=Lambda(one_hot_Input)(y)
        # print(X.shape)
    model=Model(inputs=[X,a0,c0],outputs=outputs)
    if summary:
        model.summary()
    return model


def train(conf:Configure):
    myModel = TrainGenModel(conf)
    myModel.compile(optimizer=conf.optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
    X,Y=conf.getTrainData()
    myModel.fit(X,Y,batch_size=conf.batchsize,epochs=conf.epochs)


    return conf
def sample(conf:Configure):
    myInferModel = PredictModel(conf)
    myInferModel.save(conf.modelPath)

    p=myInferModel.predict(conf.initialSample()) #p is a list,have shape T,m,vocabSize

    T=len(p)
    _s=''
    EOF=conf.char2idx['<EOF>']
    for t in range(T):
        idx=np.argmax(p[t][0])
        if idx==EOF:break
        _s=_s+conf.idx2char[idx]
    print(_s)
    return _s
myconf=Configure()
train(myconf)
sample(myconf)


# myInferModel=PredictModel(myconf)

# T,vocabSize,n_a,cell,densor=myconf.getModelConfig()
# m=15
# a0=np.zeros((m,n_a))
# c0=np.zeros((m,n_a))
# x=np.zeros((m,1,vocabSize))
# p=myInferModel.predict([x,a0,c0])
# print(len(p))
# print(p[0].shape)