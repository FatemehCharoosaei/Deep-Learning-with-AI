# -*- coding: utf-8 -*-
"""

AI course examples
@author: MiladShiri
www.MiladShiri.ir


"""

import cv2
import tensorflow as tf

CAT = ['airplane', 'face']

def prepare(filepath):
    IMG_SIZE = 60
    
    im_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(im_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE,  IMG_SIZE, 1)

model = tf.keras.models.load_model('64x3_CNN.model')


filepath = 'Datasets/airplane/test/image_0203.jpg'
imagedd = [prepare(filepath)]
prediction = model.predict(imagedd)
print (CAT[int(prediction)])

filepath = 'E:/MathLab/Python/Datasets/face/test/image_0201.jpg'
prediction = model.predict([prepare(filepath)])
print (CAT[int(prediction)])
