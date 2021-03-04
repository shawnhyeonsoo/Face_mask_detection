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

masked_data_path = path + '/dataset/AFDB_masked_face_dataset/'
masked_faces = []
for files in os.listdir(masked_data_path):
    if os.path.isdir(masked_data_path+files):
        for file in os.listdir(masked_data_path+files):
            img = cv2.imread(masked_data_path+files+'/'+file,cv2.IMREAD_GRAYSCALE)
            img2 = cv2.resize(img, dsize = (60,60), interpolation = cv2.INTER_AREA)
            masked_faces.append(img2)


unmasked_faces_label = [[0,1] for i in range(len(unmasked_faces))]
#unmasked_faces_label = np.array([[0,1] for i in range(len(unmasked_faces))])
masked_faces_label = [[1,0] for i in range(len(masked_faces))]
#masked_faces_label = np.array([[1,0] for i in range(len(masked_faces))])

masked_test = list(zip(masked_faces, masked_faces_label))
unmasked_test = list(zip(unmasked_faces,unmasked_faces_label))

test = masked_test + unmasked_test

import random
random.shuffle(test)

check_shuffle = [i for i in range(len(test)) if test[i][1] == [1,0]]

data_x = np.array([test[i][0] for i in range(len(test))])
data_y = [test[i][1] for i in range(len(test))]

percentage = 0.85
splitter = int(len(test)*percentage)

train_x = data_x[:splitter]
#train_y = data_y[:splitter]

val_x = data_x[splitter:]
#val_y = data_y[splitter:]

######MODELING #############
import tensorflow as tf
from tensorflow.keras import layers, models

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

train_x = train_x.reshape((-1, 60,60,1))

label = []
for i in data_y:
    if i == [0,1]:
        label.append(0)
    else:
        label.append(1)

train_y = np.array(label[:splitter])
val_y = np.array(label[splitter:])

train_x, val_x = train_x / 255.0, val_x /255.0

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3), activation= 'relu', input_shape = (60,60,1)))
model.add(layers.Conv2D(32,(3,3),activation = 'relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(64,(3,3),activation = 'relu'))
model.add(layers.Conv2D(64,(3,3),activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(64,(3,3), activation ='relu'))
model.add(layers.Dropout(0.5))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(32,activation = 'relu'))
model.add(layers.Dense(2, activation = 'softmax'))
model.compile(optimizer = 'adam',loss= 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(train_x, train_y, epochs = 100, batch_size = 32)
val_loss, val_acc = model.evaluate(val_x, val_y, verbose = 2)
train_outputs = []
outputs = []
for j in range(len(val_x)):
    train_outputs.append(int(tf.argmax(model.predict(val_x[j:j+1]),1)))
print(val_acc)
