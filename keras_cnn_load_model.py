# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 21:22:19 2022
@author: FrankNGUEN
https://miai.vn/2020/11/06/khoa-hoc-mi-python-bai-5-python-voi-keras-cnn/
https://www.miai.vn/2020/11/06/khoa-hoc-mi-python-bai-5-python-voi-keras-cnn/
"""
#import thu vien
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import matplotlib.pyplot as plt
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
from keras.models import load_model
#-----------------------------------------------------------------------------
# load dataset:
(X_train, y_train), (X_test, y_test) = mnist.load_data()
#plt.imshow(X_train[3])
X_train1 = X_train.reshape(60000,28,28,1)
X_test1  = X_test.reshape(10000,28,28,1)
y_train1 = to_categorical(y_train)
y_test1  = to_categorical(y_test)
#-----------------------------------------------------------------------------
#load model
model      = load_model("model/cnn_model.h5")
#-----------------------------------------------------------------------------
#test:
plt.imshow(X_test[6])    
y_hat = model.predict(X_test1[6:7])
print("Xac suat anh dau vao: ",y_hat)
y_label = np.argmax(y_hat,axis=1)
print("So dau vao la so :", y_label)
