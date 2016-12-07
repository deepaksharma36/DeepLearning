import numpy as np
import os
def data_files_vec(dirNames):
    data_files=[]
    with open(dirNames) as dirFP:
        for dir_lbl in dirFP:
            dir=dir_lbl.strip().split()[0]
            lbl=int(dir_lbl.strip().split()[1])
            for filename in os.listdir(dir):
                data_files.append([dir+filename,lbl])

    data_files=np.array(data_files)
    np.random.shuffle(data_files)
    return data_files


