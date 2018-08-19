import numpy as np
from keras.layers import Concatenate,Dropout,Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model
from matplotlib.pyplot import imshow
import  matplotlib.pyplot  as plt
import keras.backend as K
from CIFAR10Utils import *
from keras.initializers import he_normal,glorot_normal
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import  ModelCheckpoint,ReduceLROnPlateau,LearningRateScheduler
K.set_image_data_format('channels_last')
K.set_learning_phase(1)


def callback():
    '''
        define ModelCheckpoint
            learnRateSchedule
            ReduceOnPlateau
        callback function
    :return:
    '''
    checkpoint = ModelCheckpoint(filename, 'val_acc', verbose=1, save_best_only=True)

    def lr_schedular(epoch, lr):
        print('Learning rate: ', lr)
        return lr

    lr_callback = LearningRateScheduler(lr_schedular)
    reducePlateau = ReduceLROnPlateau('val_loss', factor=0.9, patience=10, verbose=1)

    return [checkpoint,lr_callback,reducePlateau]
def _inception(X):
    a_7=Conv2D(filters=32,kernel_size=(1,1),activation='relu',kernel_initializer=he_normal())(X)
    a_7=Conv2D(filters=32, kernel_size=(7, 7), activation='relu',padding='SAME',kernel_initializer=he_normal())(a_7)
    a_7 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu',kernel_initializer=he_normal())(a_7)

    a_5=Conv2D(filters=32,kernel_size=(1,1),activation='relu',kernel_initializer=he_normal())(X)
    a_5=Conv2D(filters=32, kernel_size=(5, 5), activation='relu',padding='SAME',kernel_initializer=he_normal())(a_5)
    a_5 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu',kernel_initializer=he_normal())(a_5)


    a_3=Conv2D(filters=32,kernel_size=(1,1),activation='relu',kernel_initializer=he_normal())(X)
    a_3=Conv2D(filters=32, kernel_size=(3, 3), activation='relu',padding='SAME',kernel_initializer=he_normal())(a_3)
    a_3 = Conv2D(filters=96, kernel_size=(1, 1), activation='relu',kernel_initializer=he_normal())(a_3)


    a_max=Conv2D(filters=32,kernel_size=(1,1),activation='relu',kernel_initializer=he_normal())(X)
    a_max=MaxPooling2D((2,2),strides=1,padding='SAME')(a_max)
    a_max = Conv2D(filters=64, kernel_size=(1, 1), activation='relu',kernel_initializer=he_normal())(a_max)
    out=Concatenate()([a_3,a_7,a_5,a_max])
    return out

'''
wrap up the given generator
'''
def customGen(gen):
    '''
    wrap up the given generator,when call gen.next(),
    get X,Y,Y have shape [batchSize,10]

    so this wrap up function convert Y to a list of 10 element,
    Y[i] is a numpy array with shape (m,)

    :param gen: a generator return by ImageDataGenerator.flow(X,Y)
    :return:

    '''
    while True:
        X,Y=gen.next()
        Y=list(np.transpose(Y, [1, 0]))
        yield (X,Y)

def MyInception(input_shape=(64, 64, 3), classes=6,Flag='softmax'):
    X_input=Input(input_shape)

    X=Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer=he_normal())(X_input) #shape 32,32,64
    X=BatchNormalization(axis=3)(X)
    X=MaxPooling2D((2,2))(X)#shape 16,16,64

    X=Conv2D(filters=128,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer=he_normal())(X)
    X=BatchNormalization(axis=3)(X)
    X=MaxPooling2D((2,2))(X) #shape 8,8,128

    X=_inception(X)#shape 8,8,256
    X=BatchNormalization(axis=3)(X)
    X=MaxPooling2D((2,2))(X)#shape 4,4,256
    # output layer
    X = Flatten()(X)
    X = Dropout(0.2)(X)#shape 8x8x128=8192
    # X=Dense(1024,activation='relu',kernel_initializer=he_normal())(X)



    '''
        Try multi Binary classer
    '''

    if Flag=='binary':
        out=[]
        for i in range(classes):
            out.append(Dense(1,activation='sigmoid',kernel_initializer=he_normal(),name='loss'+str(i+1))(X))
    else:
        out = Dense(classes, activation='softmax', kernel_initializer=he_normal())(X)
    # Create model
    model = Model(inputs=X_input, outputs=out, name='ResNet50')
    model.summary()
    return model

FLAG='binary'
'''
Prepare the dataset
'''
X_train,Y_train,X_test,Y_test,classes=load_dataset(flaten=False)
m,batch=X_train.shape[0],64
if FLAG=='binary':
    # Y_train = list(np.transpose(Y_train, [1, 0]))
    Y_test=list(np.transpose(Y_test,[1,0]))

aug = ImageDataGenerator(width_shift_range = 0.125, height_shift_range = 0.125, horizontal_flip = True,zoom_range=0.1)
aug.fit(X_train)
gen = aug.flow(X_train, Y_train, batch_size=batch)
if FLAG=='binary':
    gen=customGen(gen)

    cc=next(gen)
'''
define the model
'''
model = MyInception(input_shape = (32, 32, 3), classes = 10,Flag=FLAG)
if FLAG=='binary':
    '''
    can write as loss=['binary_crossentropy','binary_crossentropy','binary_crossentropy'....]
    
    '''
    model.compile(optimizer='Adam',loss='binary_crossentropy', metrics=['accuracy'])
else:
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])


epoch=5000
filename='outputs/cfar10_myinception_binary.h5'

h = model.fit_generator(generator=gen,
                        epochs=epoch,
                        steps_per_epoch=m//batch,
                        validation_data=(X_test, Y_test),
                        callbacks=callback()
                        )



