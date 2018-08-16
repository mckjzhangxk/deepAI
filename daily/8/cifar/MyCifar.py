import numpy as np
from keras.layers import Concatenate,Dropout,Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model
from matplotlib.pyplot import imshow
import  matplotlib.pyplot  as plt
import keras.backend as K
from CIFAR100Utils import *
from keras.initializers import he_normal,glorot_normal
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import  ModelCheckpoint,ReduceLROnPlateau,LearningRateScheduler
K.set_image_data_format('channels_last')
K.set_learning_phase(1)



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

def MyInception(input_shape=(64, 64, 3), classes=6):
    X_input=Input(input_shape)

    X=Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer=he_normal())(X_input)
    X=BatchNormalization(axis=3)(X)
    X=MaxPooling2D((2,2))(X)

    X=Conv2D(filters=128,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer=he_normal())(X)
    X=BatchNormalization(axis=3)(X)
    X=MaxPooling2D((2,2))(X)

    X=_inception(X)
    X=BatchNormalization(axis=3)(X)
    X=MaxPooling2D((2,2))(X)
    # output layer
    X = Flatten()(X)
    X = Dropout(0.2)(X)
    # X=Dense(1024,activation='relu',kernel_initializer=he_normal())(X)
    X = Dense(classes, activation='softmax',kernel_initializer=he_normal())(X)


    # Create model
    model = Model(inputs=X_input, outputs=X, name='ResNet50')
    model.summary()
    return model

def NN(input_shape = (32, 32, 3), classes = 10):
    x = Input(shape=input_shape)
    y = x
    y = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu',
                      kernel_initializer='he_normal')(y)
    y = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu',
                      kernel_initializer='he_normal')(y)
    y = MaxPooling2D(pool_size=2, strides=2, padding='same')(y)

    y = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu',
                      kernel_initializer='he_normal')(y)
    y = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu',
                      kernel_initializer='he_normal')(y)
    y = MaxPooling2D(pool_size=2, strides=2, padding='same')(y)

    y = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu',
                      kernel_initializer='he_normal')(y)
    y = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu',
                      kernel_initializer='he_normal')(y)
    y = MaxPooling2D(pool_size=2, strides=2, padding='same')(y)

    y = Flatten()(y)
    y = Dropout(0.5)(y)
    y = Dense(units=classes, activation='softmax', kernel_initializer='he_normal')(y)

    model1 = Model(inputs=x, outputs=y, name='model1')

    # model1.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    model1.summary()
    return model1
model = MyInception(input_shape = (32, 32, 3), classes = 100)
model.compile(optimizer='Adam',loss='categorical_crossentropy', metrics=['accuracy'])
X_train,Y_train,X_test,Y_test,classes=load_dataset(flaten=False)


batch=64
epoch=5000
filename='outputs/cfar100_myinception.h5'
checkpoint=ModelCheckpoint(filename,'val_acc',1,True)
def lr_schedular(epoch,lr):
    print('Learning rate: ', lr)
    return lr
lr_callback=LearningRateScheduler(lr_schedular)
reducePlateau=ReduceLROnPlateau('val_loss',0.9,10,1)

aug = ImageDataGenerator(width_shift_range = 0.125, height_shift_range = 0.125, horizontal_flip = True,zoom_range=0.1)
aug.fit(X_train)
gen = aug.flow(X_train, Y_train, batch_size=batch)
h = model.fit_generator(generator=gen,
                        epochs=epoch,
                        validation_data=(X_test, Y_test),
                        callbacks=[checkpoint,reducePlateau,lr_callback]
                        )



