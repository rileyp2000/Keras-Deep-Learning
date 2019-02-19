from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import cv2


model = VGG16(weights='imagenet')

imgPath = 'C:\\Users\\PRLAX\\git\\ML-Python-Work\\BookExercises\\Data\\elephant.jpg'

img = image.load_img(imgPath, target_size=(224,224))

x = image.img_to_array(img)

x = np.expand_dims(x, axis=0)

x = preprocess_input(x)

african_elephant_output = model.output[:,386]
lastConvLayer = model.get_layer('block5_conv3')

grads = K.gradients(african_elephant_output, lastConvLayer.output)[0]

pooledGrads = K.mean(grads, axis=(0,1,2))

iterate = K.function([model.input],
                     [pooledGrads, lastConvLayer.output[0]])

pooledGradValue, convLayerOutputValue = iterate([x])

for i in range(512):
    convLayerOutputValue[:,:, i] *= pooledGradValue[i]

heatmap = np.mean(convLayerOutputValue, axis=-1)


heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)

img = cv2.imread(imgPath)

heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

superimposedImg =  heatmap * .4 + img

cv2.imwrite('C:\\Users\\PRLAX\\git\\ML-Python-Work\\BookExercises\\Data\\elephantImp.jpg', superimposedImg)
print('Done')