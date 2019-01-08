import os
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

ogDatasetPath = 'C:\\Users\\PRLAX\\git\\ML-Python-Work\\BookExercises\\Data\\train'

baseDir = 'C:\\Users\\PRLAX\\git\\ML-Python-Work\\BookExercises\\Data\\train\\Cat-Dog-Small'
#os.mkdir(baseDir)

#Directories for training test and validation
trainDir = os.path.join(baseDir, 'train')
#os.mkdir(trainDir)
trainCatsDir = os.path.join(trainDir, 'cats')
#os.mkdir(trainCatsDir)
trainDogsDir = os.path.join(trainDir, 'dogs')
#os.mkdir(trainDogsDir)

validationDir = os.path.join(baseDir, 'validation')
#os.mkdir(validationDir)
validationCatsDir = os.path.join(validationDir, 'cats')
#os.mkdir(validationCatsDir)
validationDogsDir = os.path.join(validationDir, 'dogs')
#os.mkdir(validationDogsDir)

testDir = os.path.join(baseDir, 'test')
#os.mkdir(testDir)
testCatsDir = os.path.join(testDir, 'cats')
#os.mkdir(testCatsDir)
testDogsDir = os.path.join(testDir, 'dogs')
#os.mkdir(testDogsDir)

"""
#Copies first 1000 cat images to training, validation, and test sets
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(ogDatasetPath, fname)
    dst = os.path.join(trainCatsDir, fname)
    shutil.copyfile(src, dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(1000,1500)]
for fname in fnames:
    src = os.path.join(ogDatasetPath, fname)
    dst = os.path.join(validationCatsDir, fname)
    shutil.copyfile(src, dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(1500,2000)]
for fname in fnames:
    src = os.path.join(ogDatasetPath, fname)
    dst = os.path.join(testCatsDir, fname)
    shutil.copyfile(src, dst)


#Copies dog image to training, validation, and test sets
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(ogDatasetPath, fname)
    dst = os.path.join(trainDogsDir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000,1500)]
for fname in fnames:
    src = os.path.join(ogDatasetPath, fname)
    dst = os.path.join(validationDogsDir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1500,2000)]
for fname in fnames:
    src = os.path.join(ogDatasetPath, fname)
    dst = os.path.join(testDogsDir, fname)
    shutil.copyfile(src, dst)
"""


#processing the image into usable tensors
trainDataGen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=.2,
    height_shift_range=.2,
    shear_range=.2,
    zoom_range=.2,
    horizontal_flip=True)
testDataGen = ImageDataGenerator(rescale=1./255)

trainGenerator = trainDataGen.flow_from_directory(
    trainDir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')

validationGenerator = testDataGen.flow_from_directory(
    validationDir,
    target_size=(150,150),
    batch_size=32,
    class_mode='binary')




#Sets up the convolutional network, input shape shrinks as it goes in
model = models.Sequential()
model.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

#Adds a classifier of densely connected networks that uses vectors (similar to earlier projects)
#first must flatten output of last 3d layer
model.add(layers.Flatten())
model.add(layers.Dropout(.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
              loss='binary_crossentropy',
              metrics=['acc'])

#Combines the batch generator with the compiled model
history = model.fit_generator(
    trainGenerator,
    steps_per_epoch=100,
    epochs=100,
    validation_data=validationGenerator,
    validation_steps=50)

model.save('catsAndDogsSmall1.h5')

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
