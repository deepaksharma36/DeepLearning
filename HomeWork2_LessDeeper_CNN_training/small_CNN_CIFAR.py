from keras.models import Sequential
import tensorflow as tf

tf.python.control_flow_ops = tf
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, \
    AveragePooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
import cv2
import numpy as np
from keras.utils import np_utils
import pickle
import matplotlib.pyplot as plt

BATCH_SIZE = 34
LEARNING_RATE = 0.01
REGULARIZATION_CONST = 0
MOMENTUM = 0.7
CLASSES = 10
EPOCHES = 7
EPOCHES_DONE = 0
R_mean = 0
G_mean = 0
B_mean = 0


def unpickle(file):
    """
    Method for importing the data from IFAR10 dataset
    author IFAR10 publisher Rodrigo Benenson
    https://www.cs.toronto.edu/~kriz/cifar.html \cite{CIFAR}
    :param file: Name of the file
    :return: Dictionary containing data and label
    """
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin-1')
    fo.close()
    return dict


def load_data(file, vector_type):
    """
    Extract attributes and label of the data from a given file
    :param file: Name of the file
    :param vector_type: Choice of the user
    if True return 1*3072 shape col vector
    else return 3*32*32 shape matrix
    :return:
    """
    dataset = unpickle(file)
    if not vector_type:
        # reshaping the vector
        attribute = dataset['data'].reshape(dataset['data'].shape[0], 3, 32, 32)
    else:
        attribute = dataset['data']
    label = dataset['labels']

    return attribute, label


def dataset_mean_cal():
    """
    Caculate global(entire dataset) mean for R G B colors space
    :return:
    """
    global R_mean
    global G_mean
    global B_mean
    for chuck_number in range(1, 6):
        attri, label = load_data('data_batch_' + str(chuck_number), False)
        R_mean += (np.mean(attri[:, 0, :, :]) / 6)
        G_mean += (np.mean(attri[:, 1, :, :]) / 6)
        B_mean += (np.mean(attri[:, 2, :, :]) / 6)
    attri, label = load_data('test_batch', False)
    R_mean += (np.mean(attri[:, 0, :, :]) / 6)
    G_mean += (np.mean(attri[:, 1, :, :]) / 6)
    B_mean += (np.mean(attri[:, 2, :, :]) / 6)


def data_prepration(chuck_number, train):
    """
    Preprocess the data before passing to training process
    :param chuck_number:
    :param train:
    :return:
    """
    if train:
        attri, label = load_data('data_batch_' + str(chuck_number), False)
    else:
        attri, label = load_data('test_batch', False)
    # print(attri.shape)

    attri = attri.astype('float32')
    attri = attri / 255
    # Normalizing data with global mean
    for counter in range(attri.shape[0]):
        attri[counter][0] = attri[counter][0] - R_mean / 255.0
        attri[counter][1] = attri[counter][1] - G_mean / 255.0
        attri[counter][2] = attri[counter][2] - B_mean / 255.0
        # attri[counter]=attri[counter]-np.mean(attri[counter])
    # print(np.max(attri[counter][2]),np.min(attri[counter][2]),
    # np.mean(attri[counter][2]))
    label = np.array(label)
    label = np_utils.to_categorical(np.array(label), CLASSES)
    return attri, label


def CNN_NBN(weights=None):
    """
    Implementation of a small CNN with 6 covolution layers
    batch Normalization layers has been added to the network
    after each passing the covolution layer response from relu layer
    :param weights: if requires, then weights generated from the
    previous training/epochs can be loaded.
    I used this for checking test accuracy after particular epoch
    :return: Model  and ref to first convolution layer
    """
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 32, 32)))
    model.add(Convolution2D(64, 3, 3, activation='relu', init="glorot_normal"))
    model.add(BatchNormalization(epsilon=1e-05, mode=0, axis=1, momentum=0.9))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', init="glorot_normal"))
    model.add(BatchNormalization(epsilon=1e-05, mode=0, axis=1, momentum=0.9))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', init="glorot_normal"))
    model.add(BatchNormalization(epsilon=1e-05, mode=0, axis=1, momentum=0.9))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', init="glorot_normal"))
    model.add(BatchNormalization(epsilon=1e-05, mode=0, axis=1, momentum=0.9))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', init="glorot_normal"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', init="glorot_normal"))
    model.add(BatchNormalization(epsilon=1e-05, mode=0, axis=1, momentum=0.9))
    model.add(AveragePooling2D((2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu', init="glorot_normal"))
    model.add(Dense(10, activation='softmax', init="glorot_normal"))
    if weights != None:
        print("Weight loaded:", weights)
        model.load_weights(weights)

    # Mini batch stochastic gradient decent
    # algorithm with content learning rate .01, batch size 32 and monument .7
    # for first 15 epochs and then used learning rate of learning rate .001
    # for next 5 epochs
    sgd = SGD(lr=LEARNING_RATE, momentum=MOMENTUM, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    conv_1 = None
    for layer in model.layers:
        if layer.name == 'conv_1':
            conv_1 = layer
    return model, conv_1


def CNN_BN(weights=None):
    """
    Implementation of a small CNN with 3 covolution layers
    A batch Normalization layer has been added to the network
    after each passing the covolution layer response from relu layer
    :param weights: if requires, then weights generated from the
    previous training/epochs can be loaded.
    I used this for checking test accuracy after particular epoch
    :return: Model  and ref to first convolution layer
    """
    model = Sequential()
    model.add(ZeroPadding2D((3, 3), input_shape=(3, 32, 32), ))
    conv_lyr = Convolution2D(64, 7, 7, activation='relu',
                             init="glorot_normal", name="conv_1")
    model.add(conv_lyr)
    model.add(BatchNormalization(epsilon=1e-05, mode=0, axis=1, momentum=0.9))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', init="glorot_normal"))
    model.add(BatchNormalization(epsilon=1e-05, mode=0, axis=1, momentum=0.9))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', init="glorot_normal"))
    model.add(BatchNormalization(epsilon=1e-05, mode=0, axis=1, momentum=0.9))
    model.add(AveragePooling2D((2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax', init="glorot_normal"))

    if weights != None:
        print("Weight loaded:", weights)
        model.load_weights(weights)
    # Mini batch stochastic gradient decent
    # algorithm with content learning rate .01, batch size 32
    # and monument .7 for first 30
    # epochs and then used learning rate of learning rate .001
    # for next 10 epochs
    sgd = SGD(lr=LEARNING_RATE, momentum=MOMENTUM, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    conv_1 = None
    for layer in model.layers:
        if layer.name == 'conv_1':
            conv_1 = layer
    return model, conv_1


def CNN_more_Deep(weights=None):
    """
    Implementation of a small CNN with 6 covolution layers
    batch Normalization layers has been added to the network
    after each passing the covolution layer response from relu layer
    :param weights: if requires, then weights generated from the
    previous training/epochs can be loaded.
    I used this for checking test accuracy after particular epoch
    :return: Model  and ref to first convolution layer
    """
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 32, 32)))
    model.add(Convolution2D(64, 3, 3, activation='relu', init="glorot_normal"))
    model.add(BatchNormalization(epsilon=1e-05, mode=0, axis=1, momentum=0.9))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', init="glorot_normal"))
    model.add(BatchNormalization(epsilon=1e-05, mode=0, axis=1, momentum=0.9))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', init="glorot_normal"))
    model.add(BatchNormalization(epsilon=1e-05, mode=0, axis=1, momentum=0.9))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', init="glorot_normal"))
    model.add(BatchNormalization(epsilon=1e-05, mode=0, axis=1, momentum=0.9))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', init="glorot_normal"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', init="glorot_normal"))
    model.add(BatchNormalization(epsilon=1e-05, mode=0, axis=1, momentum=0.9))
    model.add(AveragePooling2D((2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu', init="glorot_normal"))
    model.add(Dense(10, activation='softmax', init="glorot_normal"))
    if weights != None:
        print("Weight loaded:", weights)
        model.load_weights(weights)

    # Mini batch stochastic gradient decent
    # algorithm with content learning rate .01, batch size 32 and monument .7
    # for first 15 epochs and then used learning rate of learning rate .001
    # for next 5 epochs
    sgd = SGD(lr=LEARNING_RATE, momentum=MOMENTUM, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    conv_1 = None
    for layer in model.layers:
        if layer.name == 'conv_1':
            conv_1 = layer
    return model, conv_1


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


def training(model):
    """
    Perform trianing on given model
    :param model: model for training
    :return:
    """
    for epoch in range(EPOCHES):
        print("Epoch Number ", epoch + 1)
        for data_chucke in range(1, 6):  # set 1-6
            print("Data betch:", data_chucke)
            X_train, Y_train = data_prepration(data_chucke, True)
            history = (
            model.fit(X_train, Y_train, batch_size=BATCH_SIZE, nb_epoch=1,
                      shuffle=True))
            save_model(model, str(epoch + EPOCHES_DONE))
            save_train_history(history.history['loss'], history.history['acc'])
        history_backup()
        # if(epoch+EPOCHES_DONE<5 or (epoch+EPOCHES_DONE)%5==0 ):
        # test_accurcy_meansurments(str(epoch+EPOCHES_DONE))
    return model


def save_model(model, epoch):
    """
    for saving the weights and model
    code has been taken from
    https://blog.rescale.com/neural-networks-using-keras-on-rescale/
    \cite{saveModel}
    :param model:
    :return:
    """
    model.save_weights('smallCNN_CIFAR_' + epoch + '.h5', overwrite=True)


def save_train_history(new_history, new_accuracy):
    """
    Saving the results received after each epoch
    :param new_history:
    :param new_accuracy:
    :return:
    """

    with open('small_cnn_train_hist.pickle', 'rb') as f:
        record = pickle.load(f)
        record['history'].append(new_history)
        record['accuracy'].append(new_accuracy)
    with open('small_cnn_train_hist.pickle', 'wb') as f:
        # print("record saved: ",record)
        pickle.dump(record, f)


def setup_History():
    """
    Create a pickel item for hosting iterative backup
    :return:
    """
    record = {}
    record['history'] = []
    record['accuracy'] = []
    with open('small_cnn_train_hist.pickle', 'wb') as f:
        pickle.dump(record, f)


def loss_accuracy_vs_epoch_curve(fig1, loss_matrix_training,
                                 accuracy_matrix_training,
                                 loss_matrix_testing=None,
                                 accuracy_matrix_testing=None,
                                 L1='training', L2='testing'):
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
    loss__curve.plot(loss_matrix_training, label=L1)
    loss__curve.plot(loss_matrix_testing, label=L2)
    # loss__curve.text(1,1,"BATCH SIZE: "+str(BATCH_SIZE)+"\
    # nLEARNING RATE: "+str(LEARNING_RATE)+"\n "
    #            "REGULARIZATION_CONST: "+str(REGULARIZATION_CONST)+"
    # \nMOMENTUM: "+ str(MOMENTUM))
    loss__curve.set_title("Cross Entropy Loss")

    loss__curve.set_xlabel("Epochs count")
    loss__curve.set_ylabel(" Loss")
    loss__curve.legend(loc='upper left')

    accuracy__curve = fig1.add_subplot(122)

    accuracy__curve.plot(accuracy_matrix_training, label=L1)
    accuracy__curve.plot(accuracy_matrix_testing, label=L2)
    accuracy__curve.set_title("moving Avarage Accuracy")
    accuracy__curve.set_xlabel("Epochs count")
    accuracy__curve.set_ylabel("Accuracy")
    accuracy__curve.legend(loc='lower right')

    return fig1


def plot_training_History():
    """
    Open saved results of training and plot loss accuracy curves
    :return:
    """
    with open('small_cnn_train_hist.pickle', 'rb') as f:
        record = pickle.load(f)
    print(record)
    merged_record = {'history': [], 'accuracy': []}
    for counter in range(0, len(record['history']) - len(record['history']) % 5,
                         5):
        merged_record['history'].append(record['history'][counter])
        merged_record['accuracy'].append(record['accuracy'][counter])
    loss_accuracy_fig = plt.figure()
    loss_accuracy_vs_epoch_curve(loss_accuracy_fig, merged_record['history'],
                                 merged_record['accuracy'])
    plt.show()


def history_backup():
    """
    Create backup of the results for avoiding any arbitary failures
    :return:
    """

    with open('small_cnn_train_hist.pickle', 'rb') as f:
        record = pickle.load(f)

    with open('small_cnn_train_hist_backup.pickle', 'wb') as backup:
        pickle.dump(record, backup)
        print("history backup done ")


def test_accurcy_meansurments(epoch):
    """
    perfrom testing of the  model after training
    :param epoch: load weights generated after performing epoch
    :return:
    """
    model, conv_1 = small_CNN('smallCNN_CIFAR_' + epoch + '.h5')
    data_test, lbl_test = data_prepration(None, False)
    # print(model.evaluate(data_test,lbl_test, verbose=1))
    print(get_mean_per_class_acc(model, data_test, lbl_test))


def get_filters(convolution_lyr):
    """
    Draw all filters weights for a given convolution layer of a CNN network
    My code for combining filter image is inspired from the code provided on
    in www.blog.keras.io/how-convolutional-neural-networks-see-the-world.html
    :param convolution_lyr: a layer of CNN
    :return: None
    """
    weights = convolution_lyr.get_weights()
    images = []
    # Normalizing weights
    for filer_count in range(weights[0].shape[0]):
        filter = weights[0][filer_count]
        image = np.zeros([filter.shape[1], filter.shape[2], filter.shape[0]])
        for c_space in range(3):
            filter[c_space] = filter[c_space] - np.mean(filter[c_space])
            acutal_range = np.max(filter[c_space]) - np.min(filter[c_space])
            filter[c_space] = (filter[c_space] - np.min(
                filter[c_space])) / acutal_range
            image[:, :, c_space] = 255 * filter[c_space]
        image = np.uint8(image)
        print(image)
        images.append(image)
    # creating comined image
    filter_width = 7
    filter_lenght = 7
    margin = 2
    fram_width = 16 * filter_width + (16 - 1) * margin
    fram_height = 4 * filter_lenght + (4 - 1) * margin
    combined_filters = np.zeros((fram_height, fram_width, 3))
    count = 0
    for Hor in range(4):
        for ver in range(16):
            combined_filters[(filter_width + margin) * Hor:
            (filter_width + margin) * Hor + filter_width,
            (filter_lenght + margin) * ver:
            (filter_lenght + margin) * ver + filter_lenght, :] \
                = images[count]
            count += 1
    combined_filters = np.uint8(combined_filters)
    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imwrite("combined_filters.png", combined_filters)
    # cv2.waitKey(0)


def main():
    """
    Main method
    :return: None
    """
    dataset_mean_cal()
    model, conv_1 = CNN_NBN()
    # model,conv_1=CNN_BN()
    # model,conv_1=CNN_more_Deep()
    training(model)
    plot_training_History()
    get_filters(conv_1)
    print(test_accurcy_meansurments(str(43)))


def BNvsNBN():
    """
    Merge records generated by two different trainings: with and without normalization
    for plotting purpose
    :return: None
    """
    with open('small_cnn_train_hist.pickle', 'rb') as f:
        record_bn = pickle.load(f)

    with open('small_cnn_train_hist_backup_NoBN.pickle', 'rb') as f:
        record_nbn = pickle.load(f)
        merged_record_bn = {'history': [], 'accuracy': []}
        for counter in range(0, len(record_bn['history']) - len(
                record_bn['history']) % 5, 5):
            merged_record_bn['history'].append(record_bn['history'][counter])
            merged_record_bn['accuracy'].append(record_bn['accuracy'][counter])
        merged_record_nbn = {'history': [], 'accuracy': []}
        for counter in range(0, len(record_bn['history']) - len(
                record_bn['history']) % 5, 5):
            merged_record_nbn['history'].append(record_nbn['history'][counter])
            merged_record_nbn['accuracy'].append(
                record_nbn['accuracy'][counter])

        loss_accuracy_fig = plt.figure()
        loss_accuracy_vs_epoch_curve(loss_accuracy_fig,
                                     merged_record_bn['history'],
                                     merged_record_bn['accuracy'],
                                     merged_record_nbn['history'],
                                     merged_record_nbn['accuracy'],
                                     'Training_BN', 'Training_NBN'
                                     )
        plt.show()


if __name__ == "__main__":
    main()
