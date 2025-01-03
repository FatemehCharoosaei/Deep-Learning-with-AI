# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle 
from tensorflow.keras.callbacks import TensorBoard
import time

#NAME = "deep_network_{}".format(int(time.time()))
#tensor_board = TensorBoard(log_dir='logs/{}'.format(NAME))
x = pickle.load(open("face_airplane_image.pickle", "rb"))
y = pickle.load(open("face_airplane_label.pickle", "rb"))

X = x/255.0

model = Sequential()

model.add(Conv2D(64, (3,3), input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))

model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

model.fit(X, y, epochs=5, batch_size=25, validation_split=0.1) #, callbacks=[tensor_board,])
model.summary()

model.save('64x3_CNN.model')




