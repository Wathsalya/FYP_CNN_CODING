import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Activation
#from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import matplotlib.pyplot as plt


# Organize data into train, valid, test dirs
os.chdir('D:\FYP\data\Sign-Language-Digits-Dataset-master')
if os.path.isdir('train/0/') is False:
    os.mkdir('train')
    os.mkdir('valid')
    os.mkdir('test')

    for i in range(0, 10):
        shutil.move(f'{i}', 'train')
        os.mkdir(f'valid/{i}')
        os.mkdir(f'test/{i}')

        valid_samples = random.sample(os.listdir(f'train/{i}'), 30)
        for j in valid_samples:
            shutil.move(f'train/{i}/{j}', f'valid/{i}')

        test_samples = random.sample(os.listdir(f'train/{i}'), 5)
        for k in test_samples:
            shutil.move(f'train/{i}/{k}', f'test/{i}')
os.chdir('../..')


train_path = 'data/Sign-Language-Digits-Dataset-master/train'
valid_path = 'data/Sign-Language-Digits-Dataset-master/valid'
test_path = 'data/Sign-Language-Digits-Dataset-master/test'

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=train_path, target_size=(224,224), batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=valid_path, target_size=(224,224), batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=test_path, target_size=(224,224), batch_size=10, shuffle=False)


mobile = tf.keras.applications.mobilenet.MobileNet()# this is functinal API


mobile.summary()

x = mobile.layers[-6].output

print(x) # will print the description on conv_pw_13_relu (ReLU)       (None, 7, 7, 1024)        0

output = Dense(units=10, activation='softmax')(x)
model = Model(inputs=mobile.input, outputs=output) # we'll be keeping the vast majority of the original MobileNet architecutre, which has 88 layers total.




for layer in model.layers[:-23]:
    layer.trainable = False

model.summary() # last 23 layers will give us a pretty decently performing model.therfore no onwards having last 23

model.compile(tf.keras.optimizers.Adam( learning_rate=0.0001),loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=train_batches,
          steps_per_epoch=len(train_batches),
          validation_data=valid_batches,
          validation_steps=len(valid_batches),
          epochs=30,
          verbose=2
)
