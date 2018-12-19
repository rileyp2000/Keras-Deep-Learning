import os, shutil
from keras import models
from keras import layers
from keras.utils.np_utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

ogDatasetPath = 'C:\\Users\\PRLAX\\git\\ML-Python-Work\\BookExercises\\Data\\train'

baseDir = 'C:\\Users\\PRLAX\\git\\ML-Python-Work\\BookExercises\\Data\\train\\Cat-Dog-Small'
os.mkdir(baseDir)

#Directories for training test and validation
trainDir = os.path.join(baseDir, 'train')
os.mkdir(trainDir)
trainCatsDir = os.path.join(trainDir, 'cats')
os.mkdir(trainCatsDir)
trainDogsDir = os.path.join(trainDir, 'dogs')
os.mkdir(trainDogsDir)

validationDir = os.path.join(baseDir, 'validation')
os.mkdir(validationDir)
validationCatsDir = os.path.join(validationDir, 'cats')
os.mkdir(validationCatsDir)
validationDogsDir = os.path.join(validationDir, 'dogs')
os.mkdir(validationDogsDir)

testDir = os.path.join(baseDir, 'test')
os.mkdir(testDir)
testCatsDir = os.path.join(testDir, 'cats')
os.mkdir(testCatsDir)
testDogsDir = os.path.join(testDir, 'dogs')
os.mkdir(testDogsDir)



