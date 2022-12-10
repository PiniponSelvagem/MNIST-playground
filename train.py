from dataset import train_images, train_labels, test_images, test_labels
from model import model

import tensorflow as tf
import numpy as np


# Train
model.fit(
    train_images,
    train_labels,
    epochs=5,
    validation_data=(test_images, test_labels)
)

model.save('models/model.h5')