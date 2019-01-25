import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras import models, layers, optimizers
import matplotlib.pyplot as plt

convBase = VGG16(weights='imagenet',include_top=False, input_shape=(150,150,3))

baseDir = 'C:\\Users\\PRLAX\\git\\ML-Python-Work\\BookExercises\\Data\\train\\Cat-Dog-Small'

trainDir = os.path.join(baseDir, 'train')
trainCatsDir = os.path.join(trainDir, 'cats')
trainDogsDir = os.path.join(trainDir, 'dogs')

validationDir = os.path.join(baseDir, 'validation')
validationCatsDir = os.path.join(validationDir, 'cats')
validationDogsDir = os.path.join(validationDir, 'dogs')

testDir = os.path.join(baseDir, 'test')
testCatsDir = os.path.join(testDir, 'cats')
testDogsDir = os.path.join(testDir, 'dogs')


model = models.Sequential()
model.add(convBase)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu', input_dim=4*4*512))
model.add(layers.Dense(1, activation='sigmoid'))

#For when you want to freeze all the layers from the model and then add your dense ones on top
#convBase.trainable = False


#For when you want to unfreeze a few top layers to have the model preform better
convBase.trainable = True

setTrainable = False
for layer in convBase.layers:
    if layer.name == 'block5_conv1':
        setTrainable = True
    if setTrainable:
        layer.trainable = True
    else:
        layer.trainable = False

trainDataGen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=.2,
    height_shift_range=.2,
    shear_range=.2,
    zoom_range=.2,
    horizontal_flip=True,
    fill_mode='nearest')

testDataGen = ImageDataGenerator(rescale=1./255)

trainGenerator = trainDataGen.flow_from_directory(
    trainDir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

validationGenerator = testDataGen.flow_from_directory(
    validationDir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary')


model.compile(optimizer=optimizers.RMSprop(lr=1e-5),
              loss='binary_crossentropy',
              metrics=['acc'])

model.save('featuresExtracted.h5')
"""
history = model.fit_generator(
    trainGenerator,
    steps_per_epoch=100,
    epochs=10,
    validation_data=validationGenerator,
    validation_steps=50)

historyDict = history.history
lossValues = historyDict['loss']
valLossValues = historyDict['val_loss']

epochs = range(1, len(lossValues) + 1)

#plots training loss with blue dots
plt.plot(epochs, lossValues, 'bo', label='Training loss')
#plots validation loss with solid blue line
plt.plot(epochs, valLossValues, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()

#Plots the validation and training accuracy
acc = historyDict['acc']
val_acc = historyDict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()"""
