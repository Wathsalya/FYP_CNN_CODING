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


mobile = tf.keras.applications.mobilenet.MobileNet()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))


def prepare_image(file):
    img_path = 'data/MobileNet-samples/'
    img = image.load_img(img_path + file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

from IPython.display import Image
Image(filename='data/MobileNet-samples/5.jpg', width=300,height=200)

preprocessed_image = prepare_image('5.jpg')
predictions = mobile.predict(preprocessed_image)

results = imagenet_utils.decode_predictions(predictions)

print(results)


#this did not work
#check https://www.tensorflow.org/api_docs/python/tf/keras/layers/GlobalAveragePooling2D
#https://colab.research.google.com/github/saptarshidatta96/saptarshi/blob/master/_notebooks/2020-09-08-Sign-Language-Prediction.ipynb#scrollTo=YLSNS9ol8quk
#check https://www.tensorflow.org/api_docs/python/tf/keras/layers/Reshape