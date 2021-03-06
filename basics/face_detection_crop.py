import cv2
import matplotlib.pyplot as plt
import cvlib as cv
import numpy as np
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
image_path = 'test.png'
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3,5)

for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y),(x+w, y+h), (255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

plt.imshow(roi_color)
plt.show()
