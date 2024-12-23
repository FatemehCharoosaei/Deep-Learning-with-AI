# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 20:46:12 2023

@author: sara
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.datasets.mnist import load_data
import tensorflow as tf 

path = 'C:/Users/sara/Desktop/ML&DM Programming/ML&DM_Chapter5_film4(MNIST_multiclass)/'
(X_train, y_train), (X_test, y_test) = load_data(path + 'mnist.npz')

plt.imshow(X_train[5], cmap='gray')
plt.show()

X_train = X_train/255.0 #chon tasvir grayscale ast va mikhaym beyne 0, 1 normalize konim
X_test = X_test/255.0#chon tasvir grayscale ast va mikhaym beyne 0, 1 normalize konim

NN = Sequential()
NN.add(Flatten())#chon inpute ma tasvir ast & mikhaym tabdil b bordar beshe
NN.add(Dense(100, activation = tf.nn.relu))
NN.add(Dense(10, activation = tf.nn.softmax))#activation ra softmax gozashtim ta max ehtemal ra az beyne output ha dar nazar begirad
NN.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# loss ra sparse_categorical_crossentropy gozashtim ta har bar yeki output ha active shavad(yeki 1 & baghi 0 shavand)

NN.fit(X_train, y_train, epochs=3)
val_loss, val_acc = NN.evaluate(X_test, y_test)
