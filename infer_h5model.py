import cv2
import numpy as np
import pandas as pd
import sys
import os
from PIL import Image
import time
import PIL.Image as pilimg
from tensorflow.keras.models import load_model
import tensorflow as tf
tf.device("/cpu:0")



def load_trained_model():
    model_name = "C:/Users/82109/Desktop/model17_Dense.h5"  # h5모델 경로 넣기
    function_model = load_model(model_name)
    print('model load 완료')

    return function_model


def img2numpy(jpg):
    img =  pilimg.open(jpg)
    resize_img = img.resize((224, 224))

    return resize_img


if __name__ == '__main__':
    test_model = load_trained_model()
    jpg = 'C:/Users/82109/Desktop/parrot.jpg'  # 이미지 경로
    img_np = np.array(img2numpy(jpg))
    print(type(img_np))
    print(img_np.shape)
    shape =img_np.shape
    img_np = img_np.reshape(-1,112,112,3)
    print(img_np.shape)
    print(time.strftime('%c', time.localtime(time.time())))
    start = time.time()

    for i in range(0,100):
        test_model.predict(img_np)# 함수안에 추론할 이미지를 >>numpy배열로 바꾸기

    print(time.strftime('%c', time.localtime(time.time())))

    t = time.time() - start
    print(t)
    print(t / 100)
