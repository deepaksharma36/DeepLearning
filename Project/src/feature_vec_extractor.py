from pre_trained_CNN import *
import numpy as np
import cv2
from keras.utils import np_utils
import pickle

PICKLE_LIMIT=1000

from dataset_Builder import *
R_mean=0
G_mean=0
B_mean=0
R_mean =121
G_mean =125
B_mean =126

def dataset_mean_cal(data_file_vec):
    global R_mean
    global G_mean
    global B_mean
    num_samples=data_file_vec.shape[0]
    for sample_num in range(num_samples):  # meta_data['files'])):
            try:
                print("sample number:", sample_num)
                image = cv2.imread(data_file_vec[sample_num][0])
                R_mean+=(np.mean(image[0,:,:]))/num_samples
                G_mean+=(np.mean(image[1,:,:]))/num_samples
                B_mean+=(np.mean(image[2,:,:]))/num_samples

            except Exception as ex:
                print("Error: ",str(ex))
    print("R_mean",R_mean)
    print("G_mean",G_mean)
    print("B_mean",B_mean)


def pre_process_image(image):
    """
    resize and Normalize the image, before passing to CNN
    :param image:
    :return:
    """

    image = cv2.resize(image, (224, 224)).astype(np.float32)
    image[:, :, 0] -= R_mean
    image[:, :, 1] -= G_mean
    image[:, :, 2] -= B_mean
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image


def get_FC_lyr_activation(CNN_model, input, layer_number):
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


def create_Feature_vec(CNN_model, layer_num,data_file_vec,offset):
    """
    Perfroming training for softmax model
    :param CNN_model:
    :param model:
    :param layers_count:
    :return:
    """

    reload = True
    if reload:
        war = input(
            "Do you really wants to create feature vector: "+str(offset)+" again? press y")

    if reload and war == 'y':
        print("Make sure your laptop is in charging condition")
        first_flag = True
        label = []
        num_samples=data_file_vec.shape[0]
        for sample_num in range(num_samples):  # meta_data['files'])):
            try:
                print("sample number:", sample_num, " With label: ",data_file_vec[sample_num][1])
                image = cv2.imread(data_file_vec[sample_num][0])
                image = pre_process_image(image)
                image_feature = get_FC_lyr_activation(CNN_model, image, layer_num)
                if first_flag:
                    data = np.array(image_feature)
                    first_flag = False
                else:
                    data = np.append(data, np.array(image_feature), axis=0)
                label.append(int(data_file_vec[sample_num][1]))
                if (sample_num+1)%PICKLE_LIMIT==0:
                    label = np_utils.to_categorical(np.array(label), 2)
                    data_write(data,label,int((sample_num+1)/PICKLE_LIMIT))
                    del data
                    first_flag = True
                    label = list([])
            except Exception as ex:
                print('error: ',str(ex))
        label = np_utils.to_categorical(np.array(label), 2)
        data_write(data,label,int((sample_num+1)/PICKLE_LIMIT)+1,offset)

def get_Feature_vec(CNN_model,image_name,layer_num):
        image = cv2.imread(image_name)
        image = pre_process_image(image)
        return  get_FC_lyr_activation(CNN_model, image, layer_num)

def data_write(data,label,pakage_num,offset):
        print("Writing FV: ",pakage_num+offset)
        with open('train_data'+str(pakage_num+offset)+'.pickle', 'wb') as f:
           pickle.dump([data, label], f)
