__author__ = 'Deepak Sharma'

"""
IMGS 789 Deep Learning for Vision Fall 2016
Homework 1
Author: Deepak Sharma (ds5930@g.rit.edu)
Implementation of the Softmax classifier Model

"""
# Importing Liberaries
import numpy as np
import matplotlib.pyplot as plt

# global constants
REGULARIZATION_CONST = .00001
LEARNING_RATE = .027
MOMENTUM = .005
BATCH_SIZE = 10
EPOCHS = 1000


def load_data(file_name):
    """
    Method for loading  dataset
    :param file_name: name of the text file
    :return: matrix of dataset
    """
    Irishdataset = np.genfromtxt(file_name, dtype='float')
    return np.array(Irishdataset)


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


def loss_cal(Probabilities, labels, Weight):
    """
    Method for calculating  cross entropy loss
    N= number of samples
    K= Number of classes
    D=Number of attributes
    :param P: Probability matrix (N*K)
    :param labels: label vector (N*K)
    :param Weight: Weight Matrix (D*K)
    :return: loss
    """

    ng_log_likelihood = -1 * np.log(
        np.sum(np.multiply(labels, Probabilities), axis=1, keepdims=True))
    cost = np.sum(ng_log_likelihood, axis=0) / np.shape(ng_log_likelihood)[0]
    regu_loss = .5 * REGULARIZATION_CONST * np.sum(np.dot(Weight.T, Weight))
    return cost + regu_loss


def get_iris_partitions(dataset):
    """
    Helping method for rendering attributes and label vectors
    :param dataset: input dataset
    :return: attributes and Label
    """
    label = dataset[:, 0]
    attributes = dataset[:, 1:3]
    return attributes, label


def gradientDissent(dataset_training, dataset_testing, num_classes,
                    get_partition, lbl_adj):
    """

    :param dataset_training:
    :param dataset_testing: Used only for measuring
    accuracy of model after each epoch
    dataset have not been used for training
    :param num_classes:
    :param get_partition: method for separating attributes and label
    :param lbl_adj: flag for adjusting argmax output
    :return:Weight Matrix, Loss and accuracy measurements
    on test and training samples for each epochs
    """
    # List for storing accuracies and loss of each epoch
    epoch_accuracy_training = []
    epoch_accuracy_testing = []
    loss_matrix_training = []
    loss_matrix_testing = []

    dW = 0  #partial derivative of Loss function with respect to Weigh and bias
    #intialization of the Weights
    Weight = .01 * np.random.randn(dataset_training.shape[1] - lbl_adj,
                                   num_classes)
    Weight = Weight - np.mean(Weight)
    testing_attribute, Y_testing = get_partition(dataset_testing)
    Label_Vector_testing = np.zeros((testing_attribute.shape[0], num_classes))
    #Creating Label Vector for test dataset
    for i in range(Y_testing.shape[0]):
        Label_Vector_testing[i][Y_testing[i] - lbl_adj] = 1
    #outer Loop for epoches
    for epoch in range(EPOCHS):
        #shuffling training dataset
        np.random.shuffle(dataset_training)
        training_attribute, training_label = get_partition(dataset_training)
        #creating label vector(N*K) for training dataset
        Label_Vector_training = np.zeros(
            (training_attribute.shape[0], num_classes))
        for i in range(training_label.shape[0]):
            Label_Vector_training[i][training_label[i] - lbl_adj] = 1
        for batch_number in range(training_attribute.shape[0] // BATCH_SIZE):
            batch_training_samples = training_attribute[
                                     batch_number * BATCH_SIZE:(
                                                                   batch_number + 1) * BATCH_SIZE,
                                     :]
            batch_labels = Label_Vector_training[BATCH_SIZE * batch_number:(
                                                                               batch_number + 1) * BATCH_SIZE]
            probabilites = soft_max(batch_training_samples, Weight)
            dW_old = dW
            error = (probabilites - batch_labels)
            dW = np.dot(batch_training_samples.T, error) / BATCH_SIZE
            dW = dW + MOMENTUM * dW_old
            Weight = Weight - (
                LEARNING_RATE * dW + REGULARIZATION_CONST * Weight)
        loss_matrix_training.append(
            loss_cal(soft_max(training_attribute, Weight),
                     Label_Vector_training, Weight))
        epoch_accuracy_training.append(
            accuracy_tester(Weight, training_attribute, training_label))
        loss_matrix_testing.append(
            loss_cal(soft_max(testing_attribute, Weight), Label_Vector_testing,
                     Weight))
        epoch_accuracy_testing.append(
            accuracy_tester(Weight, testing_attribute, Y_testing))
    return Weight, loss_matrix_training, epoch_accuracy_training, loss_matrix_testing, epoch_accuracy_testing


def accuracy_tester(Weight, X, Y_true):
    """
    Calculate mean per class accuracy for each label
    :param Weight:
    :param X: Attributes
    :param Y_true: True Labels
    :return: Mean per class accuracy
    """
    number_classes = Weight.shape[1]
    Z = np.dot(X, Weight)
    Y_predicted = np.argmax(Z, axis=1)
    # print(Y_predicted)
    sample_accuracy = np.array(Y_predicted + 1 == Y_true)
    #print(sample_accuracy)
    sample_accuracy = sample_accuracy.reshape(sample_accuracy.shape[0], 1)
    Y_true = Y_true.reshape(Y_true.shape[0], 1)
    per_class_accuracy = np.concatenate((Y_true, sample_accuracy), axis=1)
    sum = {}
    members = {}
    for item in range(1, number_classes + 1):
        sum[item] = 0
        members[item] = 0

    for item in per_class_accuracy:
        sum[item[0]] = sum[item[0]] + item[1]
        members[item[0]] += 1
    mean_per_class_accuracy = 0
    for category in members.keys():
        mean_per_class_accuracy += (sum[category] / members[category])
    mean_per_class_accuracy = mean_per_class_accuracy / len(members.keys())
    return mean_per_class_accuracy


def decision_boundry(dataset, Weight, fig2, fig_name):
    """
    Plot decision boundries for given dataset
    :param dataset: dataset
    :param Weight: Weight matrix
    :param fig2: figure
    :param fig_name: figure name
    :return: none
    """
    X, Y = get_iris_partitions(dataset)
    classes = [1, 2, 3]
    colors = ['r', 'g', 'b']
    plt.clf()
    step = .001
    # creating mashgrid for generating decision
    # boundries using combinations of attributes values
    x1_corr, x2_corr = np.meshgrid(
        np.arange(np.min(X[:, 0]), np.max(X[:, 0]), step),
        np.arange(np.min(X[:, 1]), np.max(X[:, 1]), step))
    Y_predicted = []
    for i in range(np.shape(x1_corr)[0]):
        Y_predicted.append([])
        for j in range(np.shape(x1_corr)[1]):
            Y_predicted[i].append(np.argmax(
                np.dot(np.array([x1_corr[i, j], x2_corr[i, j]]), Weight)) + 1)

    decision_plot = fig2.add_subplot(111)
    decision_plot.contourf(x1_corr, x2_corr, np.array(Y_predicted))
    decision_plot.axis('off')
    for index in classes:
        indices = Y == index
        attribute_1 = [X[i][0] for i in range(len(X[:])) if indices[i] == True]
        attribute_2 = [X[i][1] for i in range(len(X[:])) if indices[i] == True]
        decision_plot.scatter(attribute_1, attribute_2, c=colors[index - 1],
                              label=str(index))
    decision_plot.legend(loc='upper left')
    decision_plot.set_title("Decision boundary " + fig_name)
    decision_plot.set_xlabel("First Attribute")
    decision_plot.set_ylabel("Second Attribute")

    return fig2


def loss_accuracy_vs_epoch_curve(fig1, loss_matrix_training,
                                 accuracy_matrix_training, loss_matrix_testing,
                                 accuracy_matrix_testing):
    """
    Generate Loss and Accuracy curve
    :param fig1:
    :param loss_matrix_training:
    :param accuracy_matrix_training:
    :param loss_matrix_testing:
    :param accuracy_matrix_testing:
    :return: plot figures
    """
    loss__curve = fig1.add_subplot(121)
    loss__curve.plot(loss_matrix_training, label='Training')
    loss__curve.plot(loss_matrix_testing, label='Testing')
    loss__curve.text(1, 1, "BATCH SIZE: " + str(
        BATCH_SIZE) + "\nLEARNING RATE: " + str(
        LEARNING_RATE) + "\n REGULARIZATION_CONST: " + str(
        REGULARIZATION_CONST) + "\nMOMENTUM: " + str(MOMENTUM))
    loss__curve.set_title("Cross Entropy Loss")

    loss__curve.set_xlabel("Epochs count")
    loss__curve.set_ylabel(" Loss")
    loss__curve.legend()

    accuracy__curve = fig1.add_subplot(122)

    accuracy__curve.plot(accuracy_matrix_training, label='Training')
    accuracy__curve.plot(accuracy_matrix_testing, label='Testing')
    accuracy__curve.set_title("Mean Per Class Accuracy")
    accuracy__curve.set_xlabel("Epochs count")
    accuracy__curve.set_ylabel("Accuracy")
    accuracy__curve.legend(loc='upper left')

    return fig1


def main():
    """
    Driver Method
    :return:
    """
    data_file_training = 'iris-train.txt'
    data_file_testing = 'iris-test.txt'
    dataset_training = load_data(data_file_training)
    dataset_testing = load_data(data_file_testing)
    num_classes = 3
    # training the model
    Weight, loss_matrix_training, accuracy_matrix_training, loss_matrix_testing, accuracy_matrix_testing = gradientDissent(
        dataset_training, dataset_testing, num_classes,
        get_iris_partitions, 1)
    loss_accuracy_fig = plt.figure()
    decision_boundary_fig_testing = plt.figure()
    decision_boundary_fig_training = plt.figure()
    # plotting the accuracy and loss curves
    loss_accuracy_vs_epoch_curve(loss_accuracy_fig, loss_matrix_training,
                                 accuracy_matrix_training,
                                 loss_matrix_testing, accuracy_matrix_testing)
    decision_boundry(dataset_testing, Weight, decision_boundary_fig_testing,
                     "Testing")
    decision_boundry(dataset_training, Weight, decision_boundary_fig_training,
                     "Training")
    plt.show()
    plt.close('all')


if __name__ == "__main__":
    main()