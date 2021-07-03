import logging
logging.getLogger("tensorflow").setLevel(logging.DEBUG)

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pathlib
#tf.enable_eager_execution()
import tensorflow as tf
import pandas as pd
from tensorflow.keras.datasets import cifar10

tf.enable_eager_execution()

# Load MNIST dataset
dataset = cifar10
dataset_name = "cifar10"
(x_train, y_train), (x_test, y_test) = dataset.load_data()
img_rows = x_train.shape[1]
img_cols = x_train.shape[2]
batch_size = 512

if len(x_train.shape) == 4:
    img_channels = x_train.shape[3]
else:
    img_channels = 1

input_shape = (img_rows, img_cols, img_channels)
num_classes = len(np.unique(y_train))

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_channels).astype('float32') / 255.
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_channels).astype('float32') / 255.

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)


converter = tf.lite.TFLiteConverter.from_keras_model_file("./model/test_keras.h5")

tflite_model = converter.convert()
with open("test_keras.tflite", "wb") as f:  # normal tflite model
    f.write(tflite_model)


converter.optimizations = [tf.lite.Optimize.DEFAULT]

def representative_dataset_gen():  # 이게 문제
    data = tf.data.Dataset.from_tensor_slices(x_train).batch(1)
    for input_value in data.take(batch_size):
    # Get sample input data as a numpy array in a method of your choosing.
        yield [input_value]

converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.float32#tf.uint8
converter.inference_output_type = tf.uint8

tflite_model_quantized = converter.convert()

with open("test_keras_quantized.tflite", "wb") as f:  # save
    f.write(tflite_model_quantized)

interpreter = tf.lite.Interpreter(model_path="D:/Downloads/testData/testVGG16_model_quant_int8_0_pruning5.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
print(input_shape)
#input_data = np.array(np.random.random_sample(input_shape), dtype=np.uint8)
input_data = y_train[1000].astype(np.int8)

print(input_data.shape)
#input_data = np.reshape(x_test[9000].astype(np.int8),(1,96,72,3))
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.set_tensor(1, input_data)
#print(input_details[0]['index'])

interpreter.invoke()
#print(output_details[0]['index'])
# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
#output_data = interpreter.get_tensor(0)
print("output_data:",output_data)

pred = output_data
answer = np.round(np.array(y_test[9000]).flatten().tolist())
print(pred)
print(np.array(y_test[0]))