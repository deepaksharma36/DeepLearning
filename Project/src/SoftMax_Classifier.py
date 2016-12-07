from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.optimizers import SGD
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
tf.python.control_flow_ops = tf
import numpy as np
import random
LEARNING_RATE = 0.05
MOMENTUM = 0.7
EPOCHES=20
BATCH_SIZE=32

def build_softmax_model(input_vec_size, output_vec_size,weights=None):
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
    if weights!=None:
        print("Weight loaded")
        model.load_weights(weights)
    return model

def load_dataset(pick_num):
    with open('train_data'+str(pick_num)+'.pickle', 'rb') as f:
            [train_data, train_label] = pickle.load(f)
    return train_data,train_label

def train_classifier(model,Weights=None):

    setup_train_History()

    data_chucks=np.array([1,2,3,4,5,6,7,8,9,10,11,13,14])
    for epoch in range(EPOCHES):
        np.random.shuffle(data_chucks)
        for data_chuck in data_chucks:
            print("processing data chuck: ",data_chuck)

            train_data,train_label=load_dataset(data_chuck)
            history = model.fit(train_data, train_label, nb_epoch=1,
                                batch_size=BATCH_SIZE,
                                shuffle=True, validation_split=.2)

        save_train_history(history)
    save_model(model)
    return model



def test_model(model,dataset='test'):
    test_data,test_label=load_dataset(dataset)
    print(model.evaluate(test_data,test_label,
                   batch_size=32, verbose=1, sample_weight=None))

def history_plot():
    with open('train_hist.pickle', 'rb') as f:
        train_history=pickle.load(f)



    loss = train_history['loss']
    accuracy = train_history['accuracy']
    val_loss = train_history['val_loss']
    val_acc = train_history['val_acc']
    loss_accuracy_fig = plt.figure()
    loss_accuracy_vs_epoch_curve(loss_accuracy_fig, loss, accuracy, val_loss,
                                 val_acc)
    plt.show()

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
    # loss__curve.text(1, 1, "BATCH SIZE: " + str(
    #     BATCH_SIZE) + "\nLEARNING RATE: " + str(LEARNING_RATE) +
    #                  "\n "
    #                                            "REGULARIZATION_CONST: "  + "\nMOMENTUM: " + str(MOMENTUM))
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


def save_train_history(train_history):
        train_history=train_history.history
        with open('train_hist.pickle', 'rb') as f:
            record=pickle.load(f)
            record['loss'].append(train_history['loss'])
            record['accuracy'].append(train_history['acc'])
            record['val_loss'].append(train_history['val_loss'])
            record['val_acc'].append(train_history['val_acc'])

        with open('train_hist.pickle', 'wb') as f:
            print("record saved: ",record)
            pickle.dump(record, f)

def setup_train_History():
        record={}
        record['loss']=[]#.append(train_history['loss'])
        record['accuracy']=[]#.append(train_history['acc'])
        record['val_loss']=[]#.append(train_history['val_loss'])
        record['val_acc']=[]#.append(train_history['val_acc'])
        with open('train_hist.pickle', 'wb') as f:
            pickle.dump(record, f)

def select_validation_set(train_num):
    vali_num=random.randint(1,12)
    while vali_num==train_num:
        vali_num=random.randint(1,12)
    return vali_num

def save_model(model):
    """
    https://blog.rescale.com/neural-networks-using-keras-on-rescale/
    :param model:
    :return:
    """
    model.save_weights('invasive_classifier_W.h5', overwrite=True)
