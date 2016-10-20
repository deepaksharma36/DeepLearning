__author__ = 'Deepak Sharma'

"""
IMGS 789 Deep Learning for Vision Fall 2016
Homework 1
Author: Deepak Sharma (ds5930@g.rit.edu)
Implementation of the Softmax classifier Model for IFAR10 dataset
"""
import numpy as np
import itertools
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as tic
from PIL import Image

import pickle


REGULARIZATION_CONST=.00005
LEARNING_RATE=.00000005
MOMENTUM=.0000001
BATCH_SIZE=16
EPOCHS=500

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

def collect_images(per_class_limit):
    """
    A method for collecting references of the images in a dataset
    collect references of the initial occurrences of each category
    :param per_class_limit: Number of initial occurrences of each
     categories required
    :return: A dictionary(Image Box) containing references of images
    Key: Classes
    Value: List containing reference of images
    """
    train_attribute,train_label = load_data('data_batch_1',False)
    categories=max(train_label)+1
    image_box={}
    inserted=0
    for index in range(len(train_label)):
        if train_label[index] not in image_box:
            image_box[train_label[index]]=[]
        if len(image_box[train_label[index]])<per_class_limit:
                an_image = train_attribute[index]
                image_box[train_label[index]].append(an_image)
                inserted+=1
        if inserted==per_class_limit*categories:
            break
    return image_box


def display_image(image_box):
    """
    Display images one by one
    :param image_box: A dictionary containing the references of the images
    :return:None
    """

    for label_num in range(0,len(image_box.keys())):
        for image_count in range(len(image_box[label_num])):
                rgb_img=np.dstack(image_box[label_num][image_count])
                rgb_img = (rgb_img ) .astype(np.uint8)
                img=Image.fromarray(rgb_img)
                plt.imshow(img)
                plt.text(1,1,(str(label_num)+": "+str(image_count+1)))
                plt.show()

def soft_max(attribute, Weight):
    """
    Implementation of the softmax function.
    using matrix properties
    Z=(W.X)^T=X^T.W^T
    N= Number of samples
    D = number of attributes
    K = number of classes
    :param attribute(X):N*D, dimension
    :param Weight(W):D*K, dimension
    :return: Softmax vector N*K,
    """
    Z = np.dot(attribute,
               Weight)  # +bias #size of z is Num of train example * number of classes
    Z = Z - np.max(Z, axis=1, keepdims=True)
    return np.exp(Z) / np.sum(np.exp(Z), axis=1, keepdims=True)



def accuracy_tester(Weight,X,Y,number_classes):
    """
    Method for calculating Mean per class accuracy
    :param Weight: Weight matrix
    :param X:
    :param Y:
    :param number_classes:
    :return:
    """
    Z = np.dot(X,Weight)
    Y_predicted = np.argmax(Z, axis=1)
    sample_accuracy=np.array(Y_predicted == Y)
    sample_accuracy=sample_accuracy.reshape(sample_accuracy.shape[0],1)
    Y=Y.reshape(Y.shape[0],1)
    per_class_accuracy=np.concatenate((Y,sample_accuracy),axis=1)
    sum={}
    members={}
    for item in range(number_classes):
        sum[item]=0
        members[item]=0

    for item in per_class_accuracy:
        sum[item[0]]=sum[item[0]]+item[1]
        members[item[0]]+=1
    mean_per_class_accuracy=0
    for category in members.keys():
        mean_per_class_accuracy+=(sum[category]/members[category])
    mean_per_class_accuracy=mean_per_class_accuracy/len(members.keys())
    return mean_per_class_accuracy

def loss_accuracy_vs_epoch_curve(fig1,loss_matrix_training, accuracy_matrix_training,loss_matrix_testing,accuracy_matrix_testing):
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
    loss__curve.plot(loss_matrix_testing, label='Testing')
    loss__curve.text(1,1,"BATCH SIZE: "+str(BATCH_SIZE)+"\nLEARNING RATE: "+str(LEARNING_RATE)+"\n REGULARIZATION_CONST: "+str(REGULARIZATION_CONST)+"\nMOMENTUM: "+ str(MOMENTUM))
    loss__curve.set_title("Cross Entropy Loss")

    loss__curve.set_xlabel("Epochs count")
    loss__curve.set_ylabel(" Loss")
    loss__curve.legend()

    accuracy__curve= fig1.add_subplot(122)

    accuracy__curve.plot(accuracy_matrix_training, label='Training')
    accuracy__curve.plot(accuracy_matrix_testing, label='Testing')
    accuracy__curve.set_title("Mean Per Class Accuracy")
    accuracy__curve.set_xlabel("Epochs count")
    accuracy__curve.set_ylabel("Accuracy")
    accuracy__curve.legend(loc='upper left')

    return fig1

def loss_cal(P,batch_labels,Weight):
    """
    Calculate cross entropy and regularization loss
    :param P: Probability matrix
    :param batch_labels:
    :param Weight:
    :return:
    """
    ng_log_likelihood=-1*np.log((np.sum(np.multiply(P,batch_labels),axis=1,keepdims=True)))
    cost=np.sum(ng_log_likelihood, axis=0)/np.shape(ng_log_likelihood)[0]
    cost=np.log(cost)
    regu_loss=.5*REGULARIZATION_CONST*np.sum(np.dot(Weight.T,Weight))
    return cost+regu_loss

def gradientDissent( num_classes,X_testing,Y_testing,X_training,Y_training,Weight,epoch_accuracy_training,\
                     epoch_accuracy_testing,loss_matrix_training,loss_matrix_testing ):
    """
    gradient dissent Algorithm Implementation
    :param num_classes:
    :param X_testing:
    :param Y_testing:
    :param X_training:
    :param Y_training:
    :param Weight:
    :param epoch_accuracy_training:
    :param epoch_accuracy_testing:
    :param loss_matrix_training:
    :param loss_matrix_testing:
    :return:
    """
    dW=0
    Label_Vector_testing=np.zeros((X_testing.shape[0],num_classes))
    for i in range(Y_testing.shape[0]):
        Label_Vector_testing[i][Y_testing[i]]=1
    for epoch in range(EPOCHS):
        Label_Vector_training=np.zeros((X_training.shape[0],num_classes))
        for i in range(Y_training.shape[0]):
            Label_Vector_training[i][Y_training[i]]=1
        training_sample=X_training
        for batch_number in range(training_sample.shape[0]//BATCH_SIZE):
            batch_training_samples=X_training[batch_number*BATCH_SIZE:(batch_number+1)*BATCH_SIZE,:]
            batch_labels=Label_Vector_training[BATCH_SIZE*batch_number:(batch_number+1)*BATCH_SIZE,:]
            P = soft_max(batch_training_samples,Weight)
            dW_old=dW
            error=(P-batch_labels)
            dW = np.dot(batch_training_samples.T,error )/BATCH_SIZE
            dW=dW+MOMENTUM*dW_old
            Weight =Weight-(LEARNING_RATE*dW+REGULARIZATION_CONST*Weight)
        loss_matrix_training.append(loss_cal(soft_max(X_training,Weight),Label_Vector_training,Weight))
        epoch_accuracy_training.append(accuracy_tester(Weight,X_training,Y_training,num_classes))
        loss_matrix_testing.append(loss_cal(soft_max(X_testing,Weight),Label_Vector_testing,Weight))
        epoch_accuracy_testing.append(accuracy_tester(Weight,X_testing,Y_testing,num_classes))
    return Weight,loss_matrix_training,epoch_accuracy_training,loss_matrix_testing,epoch_accuracy_testing

def training():
    """
    A method for conducting training process
    on IFAR10 dataset
    :return:None
    """
    training_dataset_count=5
    num_classes=10
    flag=True
    Weight =None#
    epoch_accuracy_training=[]
    epoch_accuracy_testing=[]
    loss_matrix_training=[]
    loss_matrix_testing=[]
    test_attri,test_label=load_data('test_batch',True)
    # adding X0 to dataset for bias
    x0=np.ones((test_attri.shape[0],1),dtype=np.int)
    test_attri=np.column_stack((x0,test_attri))
    test_label=np.array(test_label)
    train_attri=None
    train_label=None
    #reading files of the training dataset and stacking matrixis
    for training_round in range(training_dataset_count):
        attri,label= load_data('data_batch_'+str(training_dataset_count),True)
        if flag:
            train_attri=attri[:]
            train_label=label[:]
            bias=np.ones((1,num_classes),dtype=np.int)
            Weight=.01 * np.random.randn(train_attri.shape[1],num_classes)
            #adding bias to Weight Matrix
            Weight=np.row_stack((bias,Weight))
            Weight=Weight-np.mean(Weight)
            flag=False
        else:

            train_attri=np.row_stack((train_attri,attri))
            train_label.extend(label)
    train_label=np.array(train_label)
    train_label.reshape(train_label.shape[0],1)
    #adding X0 for bias
    x0=np.ones((train_attri.shape[0],1),dtype=np.int)
    train_attri=np.column_stack((x0,train_attri))

    #invoiding gradient decent algorithm
    Weight,loss_matrix_training, accuracy_matrix_training,loss_matrix_testing,accuracy_matrix_testing=\
    gradientDissent( num_classes,test_attri,test_label,train_attri,train_label,Weight, \
    epoch_accuracy_training,epoch_accuracy_testing ,loss_matrix_training,loss_matrix_testing)

    loss_accuracy_fig = plt.figure()
    #plotting loss and accuracy
    loss_accuracy_vs_epoch_curve(loss_accuracy_fig,loss_matrix_training, accuracy_matrix_training,loss_matrix_testing,accuracy_matrix_testing)
    confusion_fig=plt.figure()
    CM=confusion_matrix(Weight,test_attri,test_label,confusion_fig)
    plt.show()


def confusion_matrix(Weight, X, Y_true, confusion_fig):
    """
    Generate Confusion Matrix
    :param Weight: Weight matrix
    :param X: Attributes
    :param Y_true: True labels
    :param confusion_fig: A figure for placing output image
    :return: 2d array 10*10 containing normalized values
    """
    # keep track number of classes for each sample
    class_counter = {}
    Z = np.dot(X, Weight)
    Y_pre = np.argmax(Z, axis=1)
    classes = list(set(Y_true))
    for label in classes:
        class_counter[label] = 0
    confusion = []
    #creating confusion matrix
    for counter in range(len(classes)):
        confusion.append([])
        for _ in range(len(classes)):
            confusion[counter].append(0)

    for counter in range((Y_true.shape[0])):
        confusion[Y_true[counter]][Y_pre[counter]] += 1
        class_counter[Y_true[counter]] += 1

    #creating a normalized confusion matrix
    confusion_array = []

    for counter in range(len(classes)):
        confusion_array.append([])
    for inn_counter in range(len(classes)):
        confusion_array[counter].append(
            confusion[counter][inn_counter] / class_counter[counter])
    confusion_fig_sub = confusion_fig.add_subplot(111)
    colors = confusion_fig_sub.matshow(confusion_array, interpolation='nearest')
    confusion_fig.colorbar(colors)
    labels = ['', 'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
              'frog', 'horse', 'ship', 'truck']
    confusion_fig_sub.set_xticklabels(labels)
    confusion_fig_sub.set_yticklabels(labels)
    confusion_fig_sub.xaxis.set_major_locator(tic.MultipleLocator(1))
    confusion_fig_sub.yaxis.set_major_locator(tic.MultipleLocator(1))
    return confusion_array



def main():
    """
    Driver Method
    :return:
    """
    limit_per_class=3
    image_box=collect_images(limit_per_class)
    display_image(image_box)
    #training()

main()