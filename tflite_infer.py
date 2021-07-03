import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
tf.enable_eager_execution()


import os
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

# No GPU found

np.random.seed = 55

# dataset
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

# model
model = load_model("./model/test_keras.h5")

import time
n =1000
start = time.time()
pred = model.predict(x_test[:n])
print("time :", (time.time()-start)/n)
#print(pred)

loss, acc = model.evaluate(x_test, y_test)
print("normal model acc :", acc)



interpreter = tf.lite.Interpreter("./model/test_keras_quantized.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]["shape"]

score = 0
#x_test = x_test.astype(np.uint8)

all_time = 0


for i in range(100):
    #print(i)
    input_data = x_test[i:i + 1]  # shape과 data간 차원수 잘 맞추기
    #input_data = x_test[i:i + 1]
    #print(input_details[0])
    interpreter.set_tensor(input_details[0]["index"], input_data)
    start = time.time()
    interpreter.invoke()
    one_time =(time.time() - start) / n
    #print("one_time",one_time )
    output_data = interpreter.get_tensor(output_details[0]['index'])
    #print(i, " predict :", output_data)
    #print(i, "answer :", np.argmax(y_test[i]))

    all_time = all_time + one_time
    if np.argmax(output_data) == np.argmax(y_test[i]):
        score += 1

print(score)
print(" predict :", np.argmax(output_data))
print("answer :", np.argmax(y_test[1]))
print("acc :", score/100)
print("all  time :",all_time)