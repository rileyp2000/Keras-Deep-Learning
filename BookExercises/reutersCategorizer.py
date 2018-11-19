from keras.datasets import reuters

from keras import models
from keras import layers
from keras.utils.np_utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

(trainData, trainLabels), (testData, testLabels) = reuters.load_data(num_words=10000)

#Encoding the integer sequences for review into a binary matrix
#Creates 10,000 dimensional vector that is all 0's except for sequence indicies
def vectorizeSequences(sequences, dimension=10000):
    #Creates an all 0 matrix of shape len(...), dimension
    results = np.zeros((len(sequences), dimension))
    #Sets specific indicies of results[i] to 1s
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

#Vectorizes the training and testing data
xTrain = vectorizeSequences(trainData)
xTest = vectorizeSequences(testData)

def toOneHot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results

#Vectorize the labels
oneHotTrainLabels = toOneHot(trainLabels)
oneHotTestLabels = toOneHot(testLabels)

oneHotTrainLabels = to_categorical(trainLabels)
oneHotTestLabels = to_categorical(testLabels)


#Set up network
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

#Configure the model with various functions, montiors accuracy
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#To monitor accuracy during training, create validation set of 1,000 samples from training data
#1st 10k
xVal = xTrain[:1000]
#last 10k
partialXTrain = xTrain[1000:]

yVal = yTrain[:1000]
partialYTrain = yTrain[1000:]

#Trains the model for 20 iterations over all the samples in batchs of 512
#Returns history object with tons of data
history = model.fit(partialXTrain,
                    partialYTrain,
                    epochs=10,
                    batch_size=512,
                    validation_data=(xVal, yVal))

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

plt.show()


