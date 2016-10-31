from keras.models import Sequential
from keras import backend as K
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, \
    ZeroPadding2D
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2, activity_l2
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.io import loadmat
from keras.utils import np_utils
import pickle
import random
import tensorflow as tf

tf.python.control_flow_ops = tf

import os

pet_train = 'PetsTrain.mat'
pet_test = 'PetsTest.mat'
BATCH_SIZE = 32
LEARNING_RATE = 0.05
REGULARIZATION_CONST = 0
MOMENTUM = 0.7
EPOCHES = 90
CLASSES = 37


def build_softmax_model(input_vec_size, output_vec_size):
    """
    defining classification model
    :param input_vec_size: Input vector size
    :param output_vec_size: output vector size
    (Number of classes)
    :return: classification model
    """
    model = Sequential()
    model.add(Dense(input_vec_size, input_dim=input_vec_size,
                    activation='relu'))
    model.add(Dense(output_vec_size, activation='softmax'))
    sgd = SGD(lr=LEARNING_RATE, momentum=MOMENTUM, nesterov=False)
    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def VGG_16(weights_path=None):
    """
    Implementation of VGG 16
    This code has been provided at
    https://gist.github.com/baraldilorenzo/8d096f48a1be4a2d660d
    along with weights by baraldilorenzo \cite{VGG16}
    :param weights_path:
    :return:
    """
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', name="feature_vector"))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model


def set_category_name(file_name):
    """
    Processing the imagenet categories file
    :param file_name:
    :return: List of categories
    """
    categories = []
    with open(file_name) as category_file:
        line = category_file.readline()
        while len(line.split(sep=" ")) > 1:
            categories.append(line.strip().split(sep=" ")[1:])
            line = category_file.readline()
    return categories


def build_CNN_model():
    """
    defining the VGG16 model's training method
    This code has been provided at
    https://gist.github.com/baraldilorenzo/8d096f48a1be4a2d660d
    along with weights by baraldilorenzo \cite{VGG16}

    :return:
    """

    model = VGG_16('vgg16_weights.h5')
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def predict_category(CNN_model, image, categories):
    """
    Prints the top 3 prediction made my pre-train CNN for an given test image
    :param model Pre-tained CNN model:
    :param image:
    :param categories a list containing discription for all class ids of IMAGENET:
    :return: None
    """
    output = CNN_model.predict(image)
    out_list = output.tolist()[0]
    out_list_sorted = out_list[:]
    out_list_sorted.sort()
    print(out_list_sorted)
    max1 = out_list_sorted[len(out_list_sorted) - 1]
    max2 = out_list_sorted[len(out_list_sorted) - 2]
    print(max2)
    max3 = out_list_sorted[len(out_list_sorted) - 3]
    print("First Predicted class " +str(max1)+
          str(out_list.index(max1)) + " :", categories[out_list.index(max1)])
    print("Second Predicted class " +str(max2)+
          str(out_list.index(max2)) + " :", categories[out_list.index(max2)])
    print("Third Predicted class " +str(max3)+
          str(out_list.index(max3)) + " :", categories[out_list.index(max3)])


def pre_process_image(image):
    """
    resize and Normalize the image, before passing to CNN
    :param image:
    :return:
    """
    image = cv2.resize(image, (224, 224)).astype(np.float32)
    image[:, :, 0] -= 103.939
    image[:, :, 1] -= 116.779
    image[:, :, 2] -= 123.68
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image


def get_Hidden_lyr_activation(CNN_model, input, layer_number):
    """
    return response from a pre tain CNN network for an input image
    :param CNN_model: Pre tained CNN model
    :param input: Input image
    :param layer_number: the layer number of CNN from which reponse
    need to be extracted
    :return: processed response of the pre tained network
    """
    get_layer_output = \
        K.function([CNN_model.layers[0].input, K.learning_phase()],
                   [CNN_model.layers[layer_number].output])
    layer_output = get_layer_output([input, 0])[0]

    # set all negative values to zero
    layer_output = layer_output.clip(min=0)
    # normalize them to unit length by dividing by the L_2 norm
    l2_norm = np.linalg.norm(layer_output, ord=2)
    # avoding numerical overflow
    if l2_norm != 0:
        layer_output = layer_output / l2_norm
    return layer_output


def sm_training(CNN_model, model, layers_count):
    """
    Perfroming training for softmax model
    :param CNN_model:
    :param model:
    :param layers_count:
    :return:
    """
    meta_data = loadmat(pet_train)
    data_picker = (meta_data['files'].shape[0])
    reload = False
    if reload:
        war = input(
            "Do you really wants to create feature vector again? press y")

    if reload and war == 'y':
        with open('train_data.pickle', 'wb') as f:
            train_data, train_label = load_dataset(CNN_model, layers_count,
                                                   pet_train, 0, data_picker)
            train_label = np_utils.to_categorical(np.array(train_label), 37)
            pickle.dump([train_data, train_label], f)
    else:
        with open('train_data.pickle', 'rb') as f:
            [train_data, train_label] = pickle.load(f)

    history = model.fit(train_data, train_label, nb_epoch=EPOCHES,
                        batch_size=BATCH_SIZE,
                        shuffle=True, validation_split=.2)
    return model, history


def load_dataset(CNN_model, layer_count, fileName, start, end):
    """
     Before training the softmax classifer, this method get feature vector from
     pre,train CNN  for entire dataset
    :param CNN_model: Pre tainined CNN model
    :param layer_count: Number of layers(it take response from second last
    layer)
    :param fileName: Dataset filename
    :param start: starting data index in dataset file
    :param end: ending data index in dataset file
    :return:
    """
    first_flag = True
    label = []
    dir = './images/'
    meta_data = loadmat(fileName)
    limit = end - start
    covered = {}
    covered[-1] = 'done'
    for item in range(limit):  # meta_data['files'])):
        try:
            print("sample number:", item)
            print("sample index", item)
            image = cv2.imread(dir + str(meta_data['files'][item][0][0]))
            image = pre_process_image(image)
            image = get_Hidden_lyr_activation(CNN_model, image, layer_count - 1)
            if first_flag:
                data = np.array(image)
                first_flag = False
            else:
                data = np.append(data, np.array(image), axis=0)
            print("sample label", meta_data['label'][item][0])
            label.append(meta_data['label'][item][0] - 1)
        except:
            print('error')
    return data, label


def get_mean_per_class_acc(model, X, Y):
    """
    Method for calculating Mean per class accuracy
    :param Weight: Weight matrix
    :param X:
    :param Y:
    :param number_classes:
    :return:
    """
    number_classes = CLASSES
    Y = np.argmax(Y, axis=1)
    Y = Y.reshape(Y.shape[0], 1)
    Y_predicted = model.predict_classes(X, batch_size=32, verbose=1)
    Y_predicted = Y_predicted.reshape(Y_predicted.shape[0], 1)
    sample_accuracy = np.array(Y_predicted == Y)
    per_class_accuracy = np.concatenate((Y, sample_accuracy), axis=1)
    currect_predicion = {}
    members = {}
    for label in range(number_classes):
        currect_predicion[label] = 0
        members[label] = 0

    for item in per_class_accuracy:
        currect_predicion[item[0]] = currect_predicion[item[0]] + item[1]
        members[item[0]] += 1
    mean_per_class_accuracy = 0
    for category in members.keys():
        mean_per_class_accuracy += (
            currect_predicion[category] / members[category])
    mean_per_class_accuracy = mean_per_class_accuracy / len(members.keys())
    return mean_per_class_accuracy


def sm_testing(CNN_model, layers_count, model=None):
    """
    perfrom testing for given model using on test dataset
    :param CNN_model: pre trained CNN for generating feature
    :param layers_count: number of layers in pre trained cnn
    :param model: input model
    :return: mean per class accuracy
    """
    meta_data = loadmat(pet_test)
    data_picker = meta_data['files'].shape[0]
    reload = False
    if reload:
        with open('test_data.pickle', 'wb') as f:
            test_data, test_label = load_dataset(CNN_model, layers_count,
                                                 pet_test, 0, 0 + data_picker)
            test_label = np_utils.to_categorical(np.array(test_label), 37)
            pickle.dump([test_data, test_label], f)
    else:
        with open('test_data.pickle', 'rb') as f:
            [test_data, test_label] = pickle.load(f)
        return get_mean_per_class_acc(model, test_data, test_label)


def __layer_output_tester(CNN_model, sm_model, layers_count):
    """
    testing method
    :param CNN_model:
    :param sm_model:
    :param layers_count:
    :return:
    """
    data, label = load_dataset(CNN_model, layers_count, pet_train, 0, 1)
    for i in range(3):
        output = get_Hidden_lyr_activation(sm_model, data, i)
        print(output.shape)
        print(np.max(output), np.min(output), np.mean(output))


def loss_accuracy_vs_epoch_curve(fig1, loss_matrix_training,
                                 accuracy_matrix_training,
                                 loss_matrix_testing=None,
                                 accuracy_matrix_testing=None):
    """
    Generate loss and accuracy curves
    :param fig1:
    :param loss_matrix_training:
    :param accuracy_matrix_training:
    :param loss_matrix_testing:
    :param accuracy_matrix_testing:
    :return:
    """
    loss__curve = fig1.add_subplot(121)
    loss__curve.plot(loss_matrix_training, label='Training')
    loss__curve.plot(loss_matrix_testing, label='Validation')
    loss__curve.text(1, 1, "BATCH SIZE: " + str(
        BATCH_SIZE) + "\nLEARNING RATE: " + str(LEARNING_RATE) +
                     "\n "
                                               "REGULARIZATION_CONST: " + str(
        REGULARIZATION_CONST) + "\nMOMENTUM: " + str(MOMENTUM))
    loss__curve.set_title("Cross Entropy Loss")

    loss__curve.set_xlabel("Epochs count")
    loss__curve.set_ylabel(" Loss")
    loss__curve.legend()

    accuracy__curve = fig1.add_subplot(122)

    accuracy__curve.plot(accuracy_matrix_training, label='Training')
    accuracy__curve.plot(accuracy_matrix_testing, label='Validation')
    accuracy__curve.set_title("Mean Per Class Accuracy")
    accuracy__curve.set_xlabel("Epochs count")
    accuracy__curve.set_ylabel("Accuracy")
    accuracy__curve.legend(loc='upper left')

    return fig1


def main():
    layers_count = 36
    CNN_model = build_CNN_model()
    image = cv2.imread('peppers.png')
    image = pre_process_image(image)
    categories = set_category_name('synset_words.txt')
    predict_category(CNN_model, image, categories)
    #sm_model = build_softmax_model(4096, 37)
    #sm_model, training_history = sm_training(CNN_model, sm_model, layers_count)
    #print(training_history)
    #history_backup(training_history.history)
    #sm_testing(CNN_model, layers_count, sm_model)
    #history_plot(training_history.history)
    #save_model(sm_model)
    #print(sm_testing(CNN_model, layers_count, sm_model))


def history_plot(train_history):
    loss = train_history['loss']
    accuracy = train_history['acc']
    val_loss = train_history['val_loss']
    val_acc = train_history['val_acc']
    loss_accuracy_fig = plt.figure()
    loss_accuracy_vs_epoch_curve(loss_accuracy_fig, loss, accuracy, val_loss,
                                 val_acc)
    plt.show()


def save_model(model):
    """
    for saving the CNN model, code has been checked from
    https://blog.rescale.com/neural-networks-using-keras-on-rescale/
    \cite{saveModel}
    :param model:
    :return:
    """
    model.save_weights('oxford_pet_weights.h5', overwrite=True)


def history_backup(record):
    """
    create backup of the training results for contengency plans
    :param record:
    :return:
    """
    with open('hist_backup.pickle', 'wb') as backup:
        print("history backup: ", record)
        pickle.dump(record, backup)


if __name__ == "__main__":
    main()
