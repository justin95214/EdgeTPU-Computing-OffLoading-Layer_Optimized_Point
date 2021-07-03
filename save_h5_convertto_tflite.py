#install tensorflow 2.2 or 1.15.0
#install tensorflow-gpu 2.2 or 1.15.0
#base on VGG16
#image size >> (96,72)
#nsml: nvcr.io/nvidia/tensorflow:20.06-tf2-py3
import pathlib


import tensorflow.keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import os
from sklearn.model_selection import train_test_split
import json
from tensorflow.python.client import device_lib


import tensorflow.keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import os
from sklearn.model_selection import train_test_split
import json
from tensorflow.python.client import device_lib
device_lib.list_local_devices()

from tensorflow.keras.datasets.cifar10 import load_data

class VGG():
    def cnn(self):
        rows, cols = 96, 72

        # x_data, y_data = load_data(rows, cols)

        # x_data = np.array(x_data)  # 462
        # y_data = np.array(y_data)
        # tf.keras.datasets.cifar10.load_data()
        x_data = np.load('D:/data/X0.npy')  # 100
        x_data = x_data[:100]
        y_data = np.load('D:/data/y0_data.npy') # 100
        y_data = y_data[:100]
        # x_data = np.reshape(x_data,(x_data.shape + (1,)))
        # y_data = np.reshape(y_data,(y_data.shape + (7,)))
        # y_data = np.expand_dims(y_data,axis=)

        print(x_data.shape)
        print(y_data.shape)

        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, stratify=y_data,
                                                            random_state=777)
        print(x_train.shape, x_test.shape, y_test.shape, y_test.shape)

        batch_size = 32
        num_classes = 41
        epochs = 1
        earlystopping = EarlyStopping(monitor="val_accuracy", patience=20)

        # input image dimensions
        img_rows, img_cols = rows, cols

        input_shape = (img_rows, img_cols, 3)

        model = Sequential()
        # block1
        model.add(Conv2D(64, (3, 3), input_shape=input_shape, padding="same"))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        # block2
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        #
        # block3
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        # block4
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        #
        model.add(Flatten())

        model.add(Dense(4096, activation="relu"))
        model.add(Dropout(0.5))
        # model.add(BatchNormalization())

        model.add(Dense(2048, activation="relu"))
        model.add(Dropout(0.5))

        #model.add(Dense(1024, activation="relu"))
        #model.add(Dropout(0.5))

        # model.add(BatchNormalization())

        model.add(Dense(num_classes, activation='softmax'))

        optimizers = tensorflow.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=tensorflow.keras.losses.binary_crossentropy,
                      optimizer=optimizers, metrics=['accuracy'])
        model.summary()

        model.save('test6.h5')
        print("save the model")
        history = model.fit(x_train, y_train,
                             batch_size=batch_size,
                             epochs=epochs,
                             verbose=1,
                             validation_data=(x_test, y_test),
                             callbacks=[earlystopping])
        score = model.predict(x_test)
#         print(x_test.shape, y_test.shape)
#         print('Test loss:', score[0])
#         print('Test accuracy:', score[1])

        import pathlib
        import tensorflow as tf
        # tf.enable_eager_execution()

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        tflite_models_dir = pathlib.Path("D:/download/CNN모델")
        tflite_models_dir.mkdir(exist_ok=True, parents=True)

        tflite_model_file = tflite_models_dir / "VGG16.tflite"
        tflite_model_file.write_bytes(tflite_model)

        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        mnist_train = np.load('D:/download/X5.npy') # 10
        mnist_train = mnist_train[:10]
        images = tf.cast(mnist_train, tf.float32) / 255.0

        mnist_ds = tf.data.Dataset.from_tensor_slices((images)).batch(1)

        def representative_data_gen():
            for input_value in mnist_ds.take(5):
                # Model has only one input so each data point has one element.
                yield [input_value]

        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

        tflite_model_quant = converter.convert()
        tflite_model_quant_file = tflite_models_dir / "testVGG16.tflite"
        tflite_model_quant_file.write_bytes(tflite_model_quant)

        interpreter = tf.lite.Interpreter(model_path='testVGG16.tflite')
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Test the model on random input data.
        input_shape = input_details[0]['shape']
        print(input_shape)
        # input_data = np.array(np.random.random_sample(input_shape), dtype=np.uint8)
        input_data = x_data[10].astype(np.int8)

        print(input_data.shape)
        input_data = np.reshape(x_data[10].astype(np.int8), (1, 96, 72, 3))

        # interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.set_tensor(1, input_data)
        print(input_details[0]['index'])
        interpreter.invoke()
        print(output_details[0]['index'])
        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = interpreter.get_tensor(output_details[0]['index'])
        # output_data = interpreter.get_tensor(0)
        print(output_data)

        pred = np.round(np.array(output_data).flatten().tolist())
        answer = np.round(np.array(y_data[10]).flatten().tolist())
        print(pred)
        print(answer)

a = VGG()
a.cnn()
