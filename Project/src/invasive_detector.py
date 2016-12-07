from  feature_vec_extractor import*
from pre_trained_CNN import*
from dataset_Builder import*
from SoftMax_Classifier import *
from Google_Street_image_downloader import *
from cropper import *
import cv2
WEIGHTS='invasive_classifier_W.h5'
G_Image_DRIVE='../G_Images_Exe/'
G_Image_DRIVE_RP='../G_Images_Exe_Cropped/'
LAST_FC_num=35
train_disk_num=13

def trainset_builder():
    traning_data_files=data_files_vec("data_dir.txt")
    save_plan(traning_data_files)
    traning_data_files=load_file()
    CNN_model=get_pre_tained_CNN_model()
    create_Feature_vec(CNN_model,LAST_FC_num,traning_data_files,train_disk_num)

def train_model():
    sm_model=build_softmax_model(4096,2)
    sm_model=train_classifier(sm_model)
    history_plot()


def get_Google_data():

    start=input('Give Starting Location').strip().split(sep=",")
    start=[float(corr) for corr in start]
    print(start)
    end=input('Give Ending Location').strip().split(sep=",")
    end=[float(corr) for corr in end]
    left=input('left camera angle')
    right=input('left camera angle')
    bunddle_downloadder([start,end],[left,right],G_Image_DRIVE)
    crop_images(G_Image_DRIVE,G_Image_DRIVE_RP)
    start_detection()

def start_detection():
        cv2.namedWindow('images', cv2.WINDOW_NORMAL)
        sm_model=build_softmax_model(4096,2,WEIGHTS)
        #test_model(sm_model)
        CNN_model=get_pre_tained_CNN_model()
        for filename in os.listdir(G_Image_DRIVE_RP):
            FV=get_Feature_vec(CNN_model,G_Image_DRIVE_RP+filename,LAST_FC_num)
            pre=sm_model.predict_classes(FV, batch_size=32, verbose=0)
            print(filename)
            print(pre)
            image=cv2.imread(G_Image_DRIVE_RP+filename)
            cv2.imshow("images",image)
            cv2.waitKey(0)


def save_plan(execution_plan):
        with open('ExecutionPlan.pickle', 'wb') as f:
           pickle.dump(execution_plan, f)
def load_file():
    with open('ExecutionPlan.pickle', 'rb') as f:
            file = pickle.load(f)
    return file


get_Google_data()
#train_model()
#execute_application()
#trainset_builder()
#start_detection()
#dataset_mean_cal(traning_data_files)
#sm_model=build_softmax_model(4096,2,WEIGHTS)
#test_model(sm_model,13)

