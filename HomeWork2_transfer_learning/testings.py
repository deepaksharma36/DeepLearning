from keras.models import Sequential
from keras import backend as K
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
import cv2
import numpy as np
from scipy.io import loadmat
from keras.utils import np_utils
import pickle
import random

BATCH_SIZE = 16
CLASSES = 10
EPOCHES = 1

def unpickle(file):
    """
    Method for importing the data from IFAR10 dataset
    author IFAR10 publisher Rodrigo Benenson
    https://www.cs.toronto.edu/~kriz/cifar.html
    :param file: Name of the file
    :return: Dictionary containing data and label
    """
    fo = open(file,'rb' )
    dict = pickle.load(fo,encoding='latin-1')
    fo.close()
    return dict

def load_data(file,vector_type):
    """
    Extract attributes and label of the data from a given file
    :param file: Name of the file
    :param vector_type: Choice of the user
    if True return 1*3072 shape col vector
    else return 3*32*32 shape matrix
    :return:
    """
    dataset=unpickle(file)
    if not vector_type:
        # reshaping the vector
        attribute = dataset['data'].reshape(dataset['data'].shape[0],  3, 32, 32)
    else:
        attribute = dataset['data']
    label=dataset['labels']
    return  attribute,label

def data_prepration(chuck_number):

    flag=True
    attri,label= load_data('data_batch_'+str(chuck_number),False)
    attri=attri.astype('float32')
    attri=attri/255
    for counter in range(attri.shape[0]):
        attri[counter]=attri[counter]-np.mean(attri[counter])

    print(attri.shape)

    label=np.array(label)
    label=np_utils.to_categorical(np.array(label),CLASSES)
    print(label[0])
    return attri,label

def small_CNN():
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,32,32)))
    model.add(Convolution2D(64, 7, 7, activation='relu'))

    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))

    model.add(AveragePooling2D((2,2), strides=(2,2)))

    #model.add(Flatten())
    #print('done')
    #model.add(Dense(256, activation='relu'))

    model.add(Dense(10, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.7, nesterov=True)

    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    return model

def training(model):
    for epoch in range(EPOCHES):
        print("epoch %d" % epoch)
        for data_chucke in range(1,6):
            print(data_chucke)
            X_train, Y_train = data_prepration(data_chucke)
            model.fit(X_train, Y_train, batch_size=BATCH_SIZE, nb_epoch=1)
    return model
def main():

    #model=small_CNN()
    #training(model)
    check=random.randint(0,9999)
    arr,lbl=data_prepration(1)
    print(np.max(arr[check]),np.min(arr[check]), np.mean(arr[check]))

if __name__ == "__main__":
    main()
