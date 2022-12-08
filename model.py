import tensorflow as tf
from keras import Sequential
from keras.layers import InputLayer, Reshape, Conv2D, MaxPooling2D, Flatten, Dense

# Define the model architecture
"""
#https://www.tensorflow.org/lite/performance/post_training_integer_quant
model = tf.keras.Sequential()[
    tf.keras.layers.InputLayer(input_shape=(28, 28)),
    tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10)
])
"""
model = Sequential()
model.add(InputLayer(input_shape=(28, 28)))
model.add(Reshape(target_shape=(28, 28, 1)))
model.add(Conv2D(filters=12, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(10))

# Train the digit classification model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
