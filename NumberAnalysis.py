from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical

import numpy as np
import matplotlib.pyplot as plt

(iTrain, lTrain), (iTest, lTest) = mnist.load_data()

#reshapes the training values so that they will fit easier into the network, 1D array of pixel values between 0 and 1
iTrain = iTrain.reshape((60000, 28, 28, 1))
iTrain = iTrain.astype('float32') / 255

iTest = iTest.reshape((10000, 28, 28, 1))
iTest = iTest.astype('float32') / 255

lTrain = to_categorical(lTrain)
lTest = to_categorical(lTest)

#Sets up the convolutional network, input shape shrinks as it goes in
model = models.Sequential()
model.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation='relu'))

#Adds a classifier of densely connected networks that uses vectors (similar to earlier projects)
#first must flatten output of last 3d layer
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(iTrain, lTrain, epochs=5, batch_size=64)

tLoss, tAcc = model.evaluate(iTest, lTest)
