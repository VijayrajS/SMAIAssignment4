import sys
import numpy as np
import pandas as pd

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.optimizers import RMSprop

epochs = 80

import os
path = 'Datasets/TomJerry/'

train_df = pd.read_csv(path+'train.csv')
x_img_names = list(train_df['image_file'])

from PIL import Image, ImageOps
from skimage import color

def img2array(file_):
    u = ImageOps.fit(Image.open(file_), (200, 200), method=Image.ANTIALIAS)
    u = color.rgb2gray(np.array(u))
    return u

print('x2')

x_train = np.array([img2array(path+'ntrain/'+x+'.jpg') for x in x_img_names])
img_rows=x_train[0].shape[0]
img_cols=x_train[0].shape[1]
x_train=x_train.reshape(x_train.shape[0],img_rows,img_cols,1)
y_train = np.array(train_df['emotion'])
print('x3')

x_train = x_train/255
y_train = keras.utils.to_categorical(y_train, 5)

print('x4')

from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Input, BatchNormalization
from keras.regularizers import l2

model = Sequential()

model.add(Dense(5, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print('x5')

_ = model.fit(x_train, y_train, epochs=epochs, verbose=1, validation_split=0.1)

score = model.evaluate(x_test, y_test, verbose=1)

