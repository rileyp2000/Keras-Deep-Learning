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

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
                gradient for the cost function C_x.  ``nabla_b`` and
                ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
                to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        #Feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b,w in zip(self.biases,self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        #backward pass
        delta = self.costDerivative(activations[-1],y) * \
            sigmoidPrime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta,activations[-2].transpose())
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.numLayers):
            z = zs[-1]
            sp = sigmoidPrime(z)
            delta = np.dot(self.weights[-l+1].transpose(),delta)*sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, testData):
        """Return the number of test inputs for which the neural
                network outputs the correct result. Note that the neural
                network's output is assumed to be the index of whichever
                neuron in the final layer has the highest activation."""
        testResults = [(np.argmax(self.feedforward(x)),y) for x,y in testData]
        return sum(int(x == y) for x,y in testResults)

    def costDerivative(self, outputActivations, y):
        """Return the vector of partial derivatives \partial C_x /
                \partial a for the output activations."""
        return (outputActivations - y)




def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoidPrime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
