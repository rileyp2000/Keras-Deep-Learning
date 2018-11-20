#Patrick Riley 11/19/18
#Regression Problem

from keras.datasets import boston_housing
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

(trainData, trainTargets), (testData, testTargets) = boston_housing.load_data()

#Feature-wise normalization:
#for feature in input:
#   subtract the mean of the feature
#   divide by standard deviation
#Centers feature around 0 and has a unit standard deviation
#NEVER use on test data
mean = trainData.mean(axis=0)
trainData -= mean
std = trainData.std(axis=0)
trainData /= std

testData -= mean
testData /= std


#Build network, method because must do multiple times
def buildModel():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(trainData.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    #No activation, linear layer, predict values in any range
    model.add(layers.Dense(1))

    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

#K fold validation
#Splits the data into K partitions, then trains on all versions of K-1 partitions, leaving the Kth for validation
#Validation score used in the average of the K validation scores

#Define number of partitions and data per partition
k = 4
numValSamples = len(trainData) // k
numEpochs = 500

allMAEHistories = []

for i in range(k):
    print('processing fold #', i)
    #Prepares validation data from partition i
    valData = trainData[i * numValSamples: (i + 1) * numValSamples]
    valTargets = trainTargets[i * numValSamples: (i + 1) * numValSamples]

    #Prepares the training data from all other partitions
    partialTrainData = np.concatenate(
        [trainData[:i * numValSamples], trainData[(i+1) * numValSamples:]],
        axis=0)
    partialTrainTargets = np.concatenate(
        [trainTargets[:i * numValSamples], trainTargets[(i+1) * numValSamples:]],
        axis=0)

    #builds the precompiled Keras Model
    model = buildModel()
    history = model.fit(partialTrainData, partialTrainTargets,
        validation_data=(valData, valTargets),epochs=numEpochs,batch_size=1,
                        verbose=0)
    maeHistory = history.history['val_mean_absolute_error']
    allMAEHistories.append(maeHistory)

avgMAEHistory = [
    np.mean([x[i] for x in allMAEHistories]) for i in range(numEpochs)]

def smooth_curve(points, factor=0.9):
  smoothed_points = []
  for point in points:
    if smoothed_points:
      previous = smoothed_points[-1]
      smoothed_points.append(previous * factor + point * (1 - factor))
    else:
      smoothed_points.append(point)
  return smoothed_points

smooth_mae_history = smooth_curve(avgMAEHistory[10:])

plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()
