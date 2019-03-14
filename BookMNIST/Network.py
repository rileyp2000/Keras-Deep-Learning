#Patrick Riley
#3/1/19

import random
import numpy as np

class Network(object):
    #initializes all the layers, weights, and biases for the network to be created
    def __init__(self,sizes):
        #Sizes is a list of the number of neurons in each layer
        self.numLayers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]

    #Returns the output of the network if a is the input
    def feedforward(self,a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a) + b)
        return a


    #Uses mini-batch stochastic gradient descent to train the network
    def SGD(self, trainingData, epochs, miniBatchSize, eta, testData=None):
        """Train the neural network using mini-batch stochastic
                gradient descent.  The ``training_data`` is a list of tuples
                ``(x, y)`` representing the training inputs and the desired
                outputs. ``eta`` is the learning rate. The other non-optional parameters are
                self-explanatory.  If ``test_data`` is provided then the
                network will be evaluated against the test data after each
                epoch, and partial progress printed out.  This is useful for
                tracking progress, but slows things down substantially."""
        trainingData = list(trainingData)
        n = len(trainingData)

        if testData:
            testData = list(testData)
            nTest = len(testData)

        for j in range(epochs):
            #Shuffles training data
            random.shuffle(trainingData)
            #Creates mini batches of the right size
            miniBatches = [
                trainingData[k:k+miniBatchSize] for k in range(0, n, miniBatchSize)
            ]
            for miniBatch in miniBatches:
                self.updateMiniBatch(miniBatch,eta)
            if testData:
                print("Epoch {} : {} / {}".format(j,self.evaluate(testData),nTest));
            else:
                print("Epoch {} complete".format(j))

    def updateMiniBatch(self, miniBatch, eta):
        """Update the network's weights and biases by applying
                gradient descent using backpropagation to a single mini batch.
                The "mini_batch" is a list of tuples "(x, y)", and "eta"
                is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in miniBatch:
            deltaNablaB, deltaNablaW = self.backprop(x,y)
            nabla_b = [nb+dnb for nb,dnb in zip(nabla_b,deltaNablaB)]
            nabla_w = [nw+dnw for nw,dnw in zip(nabla_w,deltaNablaW)]
        self.weights = [w-(eta/len(miniBatch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(miniBatch))*nb for b,nb in zip(self.biases, nabla_b)]



def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))
