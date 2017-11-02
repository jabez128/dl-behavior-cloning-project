# -*- coding: utf-8 -*-


import os
import sys
import datetime

import pandas as pd
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

dataset_dir = 'sample_data'

df = pd.read_csv(os.path.join(os.getcwd(), dataset_dir, 'driving_log.csv'))
center_img_path = df['center'].values
left_img_path = df['left'].values
right_img_path = df['right'].values
steering = df['steering'].values


images = []
targets = []

correction = 0.2

for i in center_img_path:
    im = Image.open(os.path.join(os.path.join(os.getcwd(), dataset_dir, i.strip())))
    im_crop = im.crop((0,50,320,140))
    #im_crop_resize = im_crop.resize((200, 66))
    images.append((np.array(im_crop)/255 - 0.5))

for i in steering:
    targets.append(float(i))

for i in left_img_path:
    im = Image.open(os.path.join(os.path.join(os.getcwd(), dataset_dir, i.strip())))
    im_crop = im.crop((0,50,320,140))
    #im_crop_resize = im_crop.resize((200, 66))
    images.append((np.array(im_crop)/255 - 0.5))

for i in steering:
    targets.append(float(i + correction))

for i in right_img_path:
    im = Image.open(os.path.join(os.path.join(os.getcwd(), dataset_dir, i.strip())))
    im_crop = im.crop((0,50,320,140))
    #im_crop_resize = im_crop.resize((200, 66))
    images.append((np.array(im_crop)/255 - 0.5))

for i in steering:
    targets.append(float(i - correction))


images = np.array(images)

targets = np.array(targets)

print(images.shape)
print(targets.shape)


#X_train, X_test, y_train, y_test = train_test_split(images, targets, test_size=0.2, random_state=42)

import keras.backend as K
from keras import models
from keras import layers
from keras import activations
from keras import callbacks
from keras import optimizers

K.clear_session()

model = models.Sequential()
model.add(layers.Conv2D(24, (5, 5), input_shape=(90, 320, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(36, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(48, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(50, activation='relu'))
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1))


model.compile(loss='mse', optimizer=optimizers.Adam(lr=1e-4))

model.fit(images, targets, validation_split=0.2, shuffle=True, epochs=5, batch_size=32)

model.save(os.path.join(os.getcwd(), 'models', 'bc_' + datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S') + '.h5'))




