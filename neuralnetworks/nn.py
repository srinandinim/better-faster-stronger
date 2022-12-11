import numpy as np
import os
import pickle

"""
REFERENCES TO MAKE BACKPROP:
- https://www.youtube.com/watch?v=VMj-3S1tku0
- https://www.youtube.com/watch?v=Ilg3gGewQ5U
- https://www.youtube.com/watch?v=w8yWXqWQYmU&t=1077s
- https://github.com/karpathy/micrograd
- https://medium.com/swlh/how-to-build-a-neural-network-from-scratch-b712d59ae641
- https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65
- https://medium.com/swlh/simple-neural-network-from-scratch-130b175eb1e6
"""

# FUNCTIONS AND DERIVATIVES FOR ACTIVATION/LOSS
def tanh(x):
    '''@param x: input value to be passed to tanh function'''
    return np.tanh(x)


def tanh_prime(x):
    '''@param x: input value to be passed to tanh_prime function'''
    return 1 - np.tanh(x)**2


def ReLU(x):
    '''@param x: input value to be passed to ReLU function'''
    return max(0, x)


def ReLU_prime(x):
    '''@param x : input value to be passed to ReLU Prime function'''
    return x if x >= 0 else 0


def identity(x):
    '''@param x: input to be passed to identity function'''
    return x


def identity_prime(x):
    '''@param x : input to be passed to identity prime function'''
    return 1


def mse(y_true, y_pred):
    '''
       @param y_true : true outputs
       @param y_pred : predicted outputs
    '''
    return np.mean(np.power(y_true-y_pred, 2))


def mse_prime(y_true, y_pred):
    '''
       @param y_true : true outputs
       @param y_pred : predicted outputs
    '''
    return 2*(y_pred-y_true)/y_true.size

# NEURAL NETWORK CLASS STRUCTURE
class Layer():
    def __init__(self):
        """every layer has an input and an output which can have derivatives"""
        self.input, self.output = None, None


class DenseLinear(Layer):
    def __init__(self, input_size, output_size):
        """every weights and biases for a layer is randomly initialized"""
        self.w = np.random.rand(input_size, output_size) - 0.5
        self.b = np.random.rand(1, output_size) - 0.5

    def forward(self, input):
        """multiply the input by weights and biases to get output"""
        self.input = input
        self.output = np.dot(self.input, self.w) + self.b
        return self.output

    def backward(self, output_error, lr):
        """update weights and biases based on gradients and error"""
        input_error = np.dot(output_error, self.w.T)
        weights_error = np.dot(self.input.T, output_error)
        self.w -= lr * weights_error
        self.b -= lr * output_error
        return input_error


class NonLinearity(Layer):
    def __init__(self, activation, activation_derivative):
        """add non-linearties to the network"""
        self.activation = activation
        self.activation_derivative = activation_derivative

    def forward(self, input):
        """run forward propogation on non-linearities"""
        self.input = input
        self.output = self.activation(self.input)
        return self.output

    def backward(self, output_error, learning_rate):
        """run backward propogation on non-linearties"""
        return self.activation_derivative(self.input) * output_error


class NeuralNetwork():
    def __init__(self):
        """NNs are a stock of layers whose gradients from loss are propogated backwards"""
        self.layers = []
        self.loss = None
        self.loss_derivative = None

    def choose_error(self, loss, loss_derivative):
        """select what kind of loss function to use to take with respect to gradients"""
        self.loss = loss
        self.loss_derivative = loss_derivative

    def add(self, layer):
        """similar to keras API just add ability to add layers to the Neural Network"""
        self.layers.append(layer)

    def predict(self, input_data):
        """pass input through all the layers of a model and return that as a result."""
        inputlen = len(input_data)
        result = []
        for i in range(inputlen):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)
        return result


def save_model(model, error, testerror=0, filename=f"vpartial_model"):
    """checkpoints the model so that we can reuse it again"""
    dirname = "/trainedmodels/"
    filepath = os.path.dirname(__file__) + dirname + filename
    with open(filepath + str(error) + "_" + str(testerror) + ".pkl", "wb") as file:
        pickle.dump(model, file)


def train(model, x_train, y_train, epochs, learning_rate):
    """trains the model and saves the model once the model is low enough"""
    leninput = len(x_train)
    for i in range(epochs):
        err = 0
        for j in range(leninput):
            # print(j)
            output = x_train[j]
            for layer in model.layers:
                output = layer.forward(output)
            err += model.loss(y_train[j], output)
            error = model.loss_derivative(y_train[j], output)
            for layer in reversed(model.layers):
                error = layer.backward(error, learning_rate)
        err /= leninput
        print('epoch %d/%d   error=%f' % (i+1, epochs, err))

        # save the model if it is a good model < 5 MSE
        if err <= 0.5:
            save_model(model, err)


def train_vpartial(model, x_train, y_train, x_test, y_test, epochs, learning_rate):
    """trains the model but also computes a validation error, saves if error is low"""
    leninput = len(x_train)
    for i in range(epochs):
        err = 0
        for j in range(leninput):
            # print(j)
            output = x_train[j]
            for layer in model.layers:
                output = layer.forward(output)
            err += model.loss(y_train[j], output)
            error = model.loss_derivative(y_train[j], output)
            for layer in reversed(model.layers):
                error = layer.backward(error, learning_rate)
        err /= leninput

        # COMPUTE VALIDATION ERROR
        test_error = 0
        lentestinput = len(x_test)
        for j in range(lentestinput):
            output = x_test[j]
            for layer in model.layers:
                output = layer.forward(output)
            test_error += model.loss(y_test[j], output)
        mse_testerror = test_error / lentestinput

        print('epoch %d/%d   train_error=%f   test_error=%f' %
              (i+1, epochs, err, mse_testerror))

        # save the model if it is a good model < 5 MSE
        if err <= 5 and mse_testerror <= 5:
            save_model(model, err, mse_testerror)
