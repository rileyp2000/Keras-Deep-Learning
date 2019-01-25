import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras import models, layers, optimizers
import matplotlib.pyplot as plt

#Uses a premade convolutional base to analyze the cat and dog images, this one does not augment the data (cheaper)
convBase = VGG16(weights='imagenet',include_top=False, input_shape=(150,150,3))

baseDir = 'C:\\Users\\PRLAX\\git\\ML-Python-Work\\BookExercises\\Data\\train\\Cat-Dog-Small'
trainDir = os.path.join(baseDir, 'train')
validationDir = os.path.join(baseDir, 'validation')
testDir = os.path.join(baseDir, 'test')

#Rescales the data
datagen = ImageDataGenerator(rescale=1./255)
batchSize = 20

def extractFeatures(directory, sampleCount):
    features = np.zeros(shape=(sampleCount, 4,4,512))
    labels = np.zeros(shape=(sampleCount))
    generator =  datagen.flow_from_directory(
        directory,
        target_size=(150,150),
        batch_size=batchSize,
        class_mode='binary')
    i = 0
    for inputsBatch, labelsBatch in generator:
        featuresBatch = convBase.predict(inputsBatch)
        features[i * batchSize : (i + 1) * batchSize] = featuresBatch
        labels[i * batchSize : (i + 1) * batchSize] = labelsBatch
        i += 1
        if i * batchSize >= sampleCount:
            print("Done")
            break
    return features, labels

#Extracts the features and labels from the pretrained covnet
trainFeatures, trainLabels = extractFeatures(trainDir, 2000)
validationFeatures, validationLabels = extractFeatures(validationDir, 1000)
testFeatures, testLabels = extractFeatures(testDir, 1000)

#Flattens the features with numpy
trainFeatures = np.reshape(trainFeatures, (2000, 4 * 4 * 512))
validationFeatures = np.reshape(validationFeatures, (1000, 4 * 4 * 512))
testFeatures = np.reshape(testFeatures, (1000, 4 * 4 * 512))

#Feeds flattened features to densely connected classifier with dropout regularization
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4*4*512))
model.add(layers.Dropout(.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(trainFeatures, trainLabels,
                    epochs=30,
                    batch_size=20,
                    validation_data=(validationFeatures, validationLabels))

"""historyDict = history.history
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
"""

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

