#This file will visualize the filters of a covnet using gradient descent
from keras.applications import VGG16
from keras import backend as K

model = VGG16(weights='imagenet',include_top=False)

layerName = 'block3_conv1'
filterIndex = 0

layerOutput = model.get_layer(layerName).output
loss = K.mean(layerOutput[:,:,:, filterIndex])