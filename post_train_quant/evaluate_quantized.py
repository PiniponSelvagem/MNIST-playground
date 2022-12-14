"""
python evaluate_quantized.py <index_of_the_test_image>
if no arg, then index 0 will be used
"""

import sys

import tensorflow as tf
import numpy as np

from dataset import test_images, test_labels

# Helper function to run inference on a TFLite model
def run_tflite_model(tflite_file, test_image_indices):
    global test_images

    # Initialize the interpreter
    interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    predictions = np.zeros((len(test_image_indices),), dtype=int)
    for i, test_image_index in enumerate(test_image_indices):
        test_image = test_images[test_image_index]
        test_label = test_labels[test_image_index]

        # Check if the input type is quantized, then rescale input data to uint8
        if input_details['dtype'] == np.uint8:
            input_scale, input_zero_point = input_details["quantization"]
            test_image = test_image / input_scale + input_zero_point

        test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
        interpreter.set_tensor(input_details["index"], test_image)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])[0]

        predictions[i] = output.argmax()

    return predictions


# function to print image
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
image = test_images[selected]

# print selected image
printImage(selected)

# Predict class
predictions = run_tflite_model("models/model_quantized.tflite", [selected])

# Return label data
print("value predicted: "+str(predictions[0]))