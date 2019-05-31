import tensorflow as tf
from CIFAR10Utils import load_dataset


def input_train_fn(path):
    X_train, Y_train, X_test, Y_test, classes=load_dataset(flaten=True,one_hot=False,filename=path)
    def mycifarfunc():
        ds=tf.data.Dataset.from_tensor_slices((X_train,Y_train))
        return ds
    return mycifarfunc()
