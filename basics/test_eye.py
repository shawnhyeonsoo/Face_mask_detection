import cv2
import matplotlib.pyplot as plt
import cvlib as cv
import numpy as np
import os
dir = 'datasets/with_mask/'
file_list = os.listdir('datasets/with_mask/')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
count = 0
for file in file_list:
    image_path = file
    img = cv2.imread(dir + image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.3,5)
    if len(eyes) > 0:
        count += 1
        #for (x,y,w,h) in eyes:
        #    cv2.rectangle(img, (x,y),(x+w, y+h), (255,0,0),2)
        #    roi_gray = gray[y:y+h, x:x+w]
        #    roi_color = img[y:y+h, x:x+w]

#plt.imshow(roi_color)
#plt.show()
