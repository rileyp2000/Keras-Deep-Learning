#This file creates a visual interpretation of the activations of a covnet on a sample image
from keras.preprocessing import image
from keras import models
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

imgPath = 'C:\\Users\\PRLAX\\git\\ML-Python-Work\\BookExercises\\Data\\train\\Cat-Dog-Small\\test\\cats\\cat.1700.jpg'
model = load_model('C:\\Users\\PRLAX\\git\\ML-Python-Work\\BookExercises\\catsAndDogsSmall1.h5')

img = image.load_img(imgPath, target_size=(150,150))
imgTensor = image.img_to_array(img)
imgTensor = np.expand_dims(imgTensor, axis=0)
imgTensor /= 255

print(imgTensor.shape)

#plt.imshow(imgTensor[0])


layerOutputs = [layer.output for layer in model.layers[:8]]
activationModel = models.Model(inputs=model.input, outputs=layerOutputs)

activations = activationModel.predict(imgTensor)

firstLayerActivation = activations[0]

#plt.matshow(firstLayerActivation[0, :, :, 7], cmap='viridis')
#plt.show()

layerNames = []

for layer in model.layers[:8]:
    layerNames.append(layer.name)

imagesPerRow = 16

for layerName, layerActivation in zip(layerNames, activations):
    nFeatures = layerActivation.shape[-1]

    size = layerActivation.shape[1]

    nCols = nFeatures // imagesPerRow  #Images Per Row
    displayGrid = np.zeros((size * nCols, imagesPerRow * size))

    for col in range(nCols):
        for row in range(imagesPerRow):
            channelImage = layerActivation[0, :, :, col * imagesPerRow + row]

            channelImage -= channelImage.mean()
            channelImage /= channelImage.std()
            channelImage *= 64
            channelImage += 128
            channelImage = np.clip(channelImage,0,255).astype('uint8')
            displayGrid[col * size : (col + 1) * size,
                        row * size : (row + 1) * size] = channelImage
    scale = 1. / size
    plt.figure(figsize=(scale * displayGrid.shape[1], scale * displayGrid.shape[0]))
    plt.title = layerName
    plt.grid(False)
    plt.imshow(displayGrid, aspect='auto', cmap='viridis')
plt.show()

