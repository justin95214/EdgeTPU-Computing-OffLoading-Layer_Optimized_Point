import logging
logging.getLogger("tensorflow").setLevel(logging.DEBUG)

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pathlib
#tf.enable_eager_execution()
import tensorflow as tf
import pandas as pd

tf.enable_eager_execution()

# Load MNIST dataset
mnist = keras.datasets.mnist
x =  np.load('D:/Downloads/testData/X1.npy')
y =  np.load('D:/Downloads/testData/y1_data.npy')
print(x.shape,y.shape)


converter = tf.lite.TFLiteConverter.from_keras_model_file("D:/Downloads/testData/test_pruning5.h5")
tflite_model = converter.convert()
tflite_models_dir = pathlib.Path("D:/Downloads/testData/")
tflite_models_dir.mkdir(exist_ok=True, parents=True)
tflite_model_file = tflite_models_dir / "VGG16mnist_model_1_pruning5.tflite"
tflite_model_file.write_bytes(tflite_model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
mnist_train = np.load('D:/Downloads/testData/X5.npy')
images = tf.cast(mnist_train, tf.float32) / 255.0
mnist_ds = tf.data.Dataset.from_tensor_slices((images)).batch(1)


def representative_data_gen():
  for input_value in mnist_ds.take(5):
    # Model has only one input so each data point has one element.
    yield [input_value]


converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model_quant = converter.convert()
tflite_model_quant_file = tflite_models_dir / "testVGG16_model_quant_int8_0_pruning5.tflite"
tflite_model_quant_file.write_bytes(tflite_model_quant)

interpreter = tf.lite.Interpreter(model_path="D:/Downloads/testData/testVGG16_model_quant_int8_0_pruning5.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
print(input_shape)
#input_data = np.array(np.random.random_sample(input_shape), dtype=np.uint8)
input_data = x[9000].astype(np.int8)

print(input_data.shape)
input_data = np.reshape(x[9000].astype(np.int8),(1,96,72,3))
#interpreter.set_tensor(input_details[0]['index'], input_data)
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
answer = np.round(np.array(y[9000]).flatten().tolist())
print(pred)
print(np.array(y[0]))