import tensorflow as tf
import os

from keras.datasets import mnist
from keras.utils import np_utils
from tensorflow.keras.models import load_model
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from utils import cropp, remove_int

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
print('setting up...')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2 as cv
from cv2 import GaussianBlur
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
#
# def remove_int(digit, x, y):
#     idx = (y != digit).nonzero()
#     return x[idx], y[idx]

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = remove_int(0, x_train, y_train)
# normalize data
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3)
loss, accuracy = model.evaluate(x_test, y_test)
print(accuracy)
print(loss)
model.save('/Users/darsh_mahra/PycharmProjects/numberrecognition/numberModel.h5')

# loading our own images

# first get all the images
model = load_model('numberModel.h5', compile=False)
import glob
all_imgs = glob.glob("/Users/darsh_mahra/PycharmProjects/numberrecognition/numbers/*.jpg")
# all_imgs is a list of uploaded images
for x in all_imgs:
    img = cv.imread(x)[:, :, 0]
    # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    # img = cv.filter2D(img, -1, kernel)
    img = cv.resize(img, (28, 28))
    print(img.shape) # 28,28
    img = np.array([img])
    print(img.shape) # 1, 28, 28
    prediction = model.predict(img)
    print('The result is probably: ' + str(np.argmax(prediction)))
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()
exit()

# testing pil image cropping

# img = cv.imread("/Users/darsh_mahra/PycharmProjects/numberrecognition/numbers/bad5.jpg")
# c = cropp(img, 20, 20) #cropped image
# # print(len(c.getdata()))
# # c = cv.resize(np.array(c), (28, 28))
# # print(type(c))
# # c = np.array(c)[:, :, 0] # 28, 28
# # print(c.shape)
# # c = np.array([c])
# # print(c.shape)
# # pred = model.predict(c)
# # print(np.argmax(pred))
# # print(c[0])
#
# cv.imshow('hello', c)
# cv.waitKey(0)
# #
# # p = model.predict(c)
# # print(np.argmax(pred))
