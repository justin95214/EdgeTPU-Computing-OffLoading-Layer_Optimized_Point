import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import tracemalloc
tf.enable_eager_execution()
import psutil
import numpy as np
tracemalloc.start()
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.device('/device:CPU:0')

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
num_classes = len(np.unique(y_train))
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)


model = load_model("./model/DNN_base_model.h5")
np.random.seed = 100

n =1

for exec_num in range(0, 5):
    start = time.time()
    # BEFORE code
    print(exec_num, " exec")
    # general RAM usage
    memory_usage_dict = dict(psutil.virtual_memory()._asdict())
    memory_usage_percent = memory_usage_dict['percent']
    print("BEFORE CODE: memory_usage_percent: ", memory_usage_percent, "%")
    # current process RAM usage
    pid = os.getpid()
    current_process = psutil.Process(pid)
    current_process_memory_usage_as_KB_b = current_process.memory_info()[0] / 2. ** 20
    print("BEFORE CODE: Current memory KB   : ", current_process_memory_usage_as_KB_b, " KB")


    pred = model.predict(x_test[:n])

    #print(pred)

    # AFTER  code
    memory_usage_dict = dict(psutil.virtual_memory()._asdict())
    memory_usage_percent = memory_usage_dict['percent']
    print("AFTER  CODE: memory_usage_percent: ",memory_usage_percent,"%")
    # current process RAM usage
    pid = os.getpid()
    current_process = psutil.Process(pid)
    current_process_memory_usage_as_KB_a = current_process.memory_info()[0] / 2.**20
    print("AFTER  CODE: Current memory KB   : ",current_process_memory_usage_as_KB_a," KB")

    print("time :", (time.time() - start) )
    print("usage after- before : ", current_process_memory_usage_as_KB_a - current_process_memory_usage_as_KB_b)
    print("--" * 30)
    del pred
    del  pid, memory_usage_dict, memory_usage_percent , current_process,

    #loss, acc = model.evaluate(x_test, y_test)

    #print("normal model acc :", acc)