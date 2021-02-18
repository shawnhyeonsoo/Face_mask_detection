import cv2
import matplotlib.pyplot as plt
import cvlib as cv
import numpy as np
import os
import dlib
dir = 'datasets/with_mask2/'
file_list = os.listdir('datasets/with_mask2/')
face_detector = dlib.get_frontal_face_detector()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
count = 0
for file in file_list:
    image_path = file
    img = cv2.imread(dir + image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector(img)
    if len(faces) > 0:
        count += 1
        #for (x,y,w,h) in faces:
        #    cv2.rectangle(img, (x,y),(x+w, y+h), (255,0,0),2)
        #    roi_gray = gray[y:y+h, x:x+w]
        #    roi_color = img[y:y+h, x:x+w]

#plt.imshow(roi_color)
#plt.show()
