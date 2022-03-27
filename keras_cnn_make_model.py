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
#-----------------------------------------------------------------------------
# load dataset:
#-----------------------------------------------------------------------------
(X_train, y_train), (X_test, y_test) = mnist.load_data()
#plt.imshow(X_test[5])
print("X shape: ")
print("X_train.shape = ", X_train.shape)
print("X_test.shape = ", X_test.shape)
X_train1 = X_train.reshape(60000,28,28,1)
X_test1  = X_test.reshape(10000,28,28,1)
print("X reshape: ")
print("X_train.shape = ", X_train1.shape)
print("X_test.shape = ", X_test1.shape)
y_train1 = to_categorical(y_train)
y_test1  = to_categorical(y_test)
#-----------------------------------------------------------------------------
#make model:
#-----------------------------------------------------------------------------
model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3), activation='relu',input_shape=(28,28,1),padding='same'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))
model.add(Conv2D(32, kernel_size=(3,3), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
model.fit(X_train1, y_train1,validation_data=(X_test1, y_test1),epochs=3)
#luu model
model.save("model/cnn_model.h5")    # train xong, luu lai. Load file de dung
#-----------------------------------------------------------------------------
