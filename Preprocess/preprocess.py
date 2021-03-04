import os
import cv2
import numpy as np
path = os.getcwd()
data_path = path + '/dataset/AFDB_face_dataset/'
unmasked_faces = []
for files in os.listdir(data_path):
    if os.path.isdir(data_path+files):
        for file in os.listdir(data_path+files):
            img = cv2.imread(data_path+files+'/'+file,cv2.IMREAD_GRAYSCALE)
            img2 = cv2.resize(img, dsize = (60,60), interpolation = cv2.INTER_AREA)
            unmasked_faces.append(img2)
unmasked_faces = np.array(unmasked_faces)

masked_data_path = path + '/dataset/AFDB_masked_face_dataset/'
masked_faces = []
for files in os.listdir(masked_data_path):
    if os.path.isdir(masked_data_path+files):
        for file in os.listdir(masked_data_path+files):
            img = cv2.imread(masked_data_path+files+'/'+file,cv2.IMREAD_GRAYSCALE)
            img2 = cv2.resize(img, dsize = (60,60), interpolation = cv2.INTER_AREA)
            masked_faces.append(img2)
masked_faces = np.array(masked_faces)


unmasked_faces_label = np.array([[0,1] for i in range(len(unmasked_faces))])
masked_faces_label = np.array([[1,0] for i in range(len(masked_faces))])
