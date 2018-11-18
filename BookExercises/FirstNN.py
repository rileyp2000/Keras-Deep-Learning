from keras.datasets import imdb
from keras import models
from keras import layers
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

(trainData, trainLabels), (testData, testLabels) = imdb.load_data(num_words=10000)

#Used to turn the data of Imdb review into english
def decrypt(ind):
    word_index = imdb.get_word_index()
    reverseWordIndex = dict(
        [(value, key) for (key, value) in word_index.items()])
    decodedReview = ' '.join(
        [reverseWordIndex.get(i - 3, '?') for i in trainData[ind]])

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

#Vectorizes the training and testing labels
yTrain= np.asarray(trainLabels).astype('float32')
yTest= np.asarray(testLabels).astype('float32')

#Setting up network
model = models.Sequential()
#16 = # hidden units, which is a dimension in the representation spaces of the layer
#relu activation == relu(dot(W, input) + b)
#                   relu = recrified linear unit, zeroes out negatives
#compute how likely is to be positive
model.add(layers.Dense(8, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(8, activation='relu'))
#final output layer, prod result between 0 and 1
model.add(layers.Dense(1, activation='sigmoid'))

#Configure the model with various functions, montiors accuracy
model.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['accuracy'])

#To monitor accuracy during training, create validation set of 10,000 samples from training data
#1st 10k
xVal = xTrain[:10000]
#last 10k
partialXTrain = xTrain[10000:]

yVal = yTrain[:10000]
partialYTrain = yTrain[10000:]

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

"""
network = models.sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network.fit(train_images, train_labels, epochs=5, batch_size=128)
"""

print("all done!")
