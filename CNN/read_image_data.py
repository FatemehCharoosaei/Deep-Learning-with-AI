# -*- coding: utf-8 -*-
"""

AI course examples
@author: MiladShiri
www.MiladShiri.ir


"""
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

DATADIR = "Datasets"
CATEGORIES = ['airplane/train', 'face/train']
IMG_SIZE = 60


def create_training_data():
    training_data = []
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                #plt.imshow(img_array, cmap="gray")
                #plt.show
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as E:
                print (E)
                
    return training_data

train_data = create_training_data()

random.shuffle(train_data)

x = []
y = []

for features, label in train_data:
    x.append(features)
    y.append(label)

X = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

pickle_out = open("face_airplane_image.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("face_airplane_label.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()






