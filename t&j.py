import sys
import numpy as np
import pandas as pd

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.optimizers import RMSprop

epochs = 80

# the data, split between train and test sets
import os
path = 'Datasets/TomJerry/'

train_df = pd.read_csv(path+'train.csv')
x_img_names = list(train_df['image_file'])

from PIL import Image, ImageOps
def img2array(file_):
    u = ImageOps.fit(Image.open(file_), (300, 300), method=Image.ANTIALIAS)
    return np.array(u)

x_train = np.array([img2array(path+'ntrain/'+x+'.jpg') for x in x_img_names])
y_train = np.array(train_df['emotion'])

x_train = x_train/255
y_train = keras.utils.to_categorical(y_train, 5)


model = Sequential()
model.add(Conv2D(128, kernel_size=(3, 3),  activation='relu', input_shape=(360,640,3)))
model.add(Conv2D(64, kernel_size=(3, 3),  activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(69, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

_ = model.fit(X_train, y_train, epochs=epochs, verbose=1, validation_split=0.1)

score = model.evaluate(x_test, y_test, verbose=1)

