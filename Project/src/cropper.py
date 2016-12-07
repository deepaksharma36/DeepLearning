import cv2
import os

def crop(image,destination):
    img = cv2.imread(image)
    #img = cv2.resize(img, (260, 260))
    side=min(img.shape[0],img.shape[1])
    size=int(side/2-5)
    y_counter=size
    counter=0
    while  y_counter<img.shape[0]:
        x_counter=size
        while x_counter<img.shape[1]:
            name=image.split(sep="/")[-1].split(sep=".")[0]
            crop_name=destination+ name+"_"+str(counter)+".jpg"
            crop_img = img[y_counter-size:y_counter, x_counter-size:x_counter]
            # Crop from x, y, w, h -> 100, 200, 300, 400
            # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
            cv2.imwrite(crop_name, crop_img)
            counter+=1
            x_counter+=size
        y_counter+=size

def crop_images(dataset,destination):
    for filename in os.listdir(dataset):
        print(filename)
        try:
            crop(dataset+filename,destination)
        except Exception as e:
            print(str(e))


def bunddle_cropper(source,destination):
    really=input("Do you want to reWrite DataSet, press 1")
    if really:
        #dataset="../invasive_dataset/invasive13/"
        #destination="../Invasive_dataset_128*128/invasive13/"
        crop_images(source,destination)


