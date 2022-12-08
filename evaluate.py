"""
python evaluate.py <index_of_the_test_image>
if no arg, then index 0 will be used
"""


import sys

from model import model

import tensorflow as tf
import numpy as np
from keras.models import load_model


def printImage(image):
    arr = image.tolist()
    print('\n'.join([''.join(['{:4}'.format(col) for col in row]) for row in arr]))




# check if index was parsed as arguement
selected = 0
if (len(sys.argv) > 1):
    selected = int(sys.argv[1])

# load test images
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
image = test_images[selected]

# print selected image
printImage(image)

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