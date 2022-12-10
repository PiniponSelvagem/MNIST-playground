"""
python evaluate.py <index_of_the_test_image>
if no arg, then index 0 will be used
"""


import sys

from model import model

import tensorflow as tf
import numpy as np
from keras.models import load_model


def printImage(selected):
    _mnist = tf.keras.datasets.mnist
    (_train_images, _train_labels), (_test_images, _test_labels) = _mnist.load_data()
    _image = _test_images[selected]

    arr = _image.tolist()
    print('\n'.join([''.join(['{:4}'.format(col) for col in row]) for row in arr]))




# check if index was parsed as argument
selected = 0
if (len(sys.argv) > 1):
    selected = int(sys.argv[1])

# load test images
from dataset import test_images
image = test_images[selected]

# print selected image
printImage(selected)

# Load trained model
model = load_model('models/model.h5')

# Predict class
image = np.expand_dims(image, axis=0)
predictions = model.predict(image)[0]

# Return label data
print(predictions)
predictions = predictions.tolist()
maxValue = max(predictions)
index = predictions.index(maxValue)
print("value predicted: "+str(index))