import numpy as np
import os
import pickle

# FUNCTIONS AND DERIVATIVES FOR ACTIVATION/LOSS
def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1 - np.tanh(x)**2


def ReLU(x):
    return max(0, x)


def ReLU_prime(x):
    return x if x >= 0 else 0


def tanh_prime(x):
    return 1 - np.tanh(x)**2


def identity(x):
    return x


def identity_prime(x):
    return 1


def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))


def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size


# NEURAL NETWORK CLASS STRUCTURE
class Layer():
    def __init__(self):
        self.input, self.output = None, None


class DenseLinear(Layer):
    def __init__(self, input_size, output_size):
        self.w = np.random.rand(input_size, output_size) - 0.5
        self.b = np.random.rand(1, output_size) - 0.5

    def forward(self, input):
        self.input = input
        self.output = np.dot(self.input, self.w) + self.b

        return self.output

    def backward(self, output_error, lr):
        input_error = np.dot(output_error, self.w.T)
        weights_error = np.dot(self.input.T, output_error)

        self.w -= lr * weights_error
        self.b -= lr * output_error

        return input_error


class NonLinearity(Layer):
    def __init__(self, activation, activation_derivative):
        self.activation = activation
        self.activation_derivative = activation_derivative

    def forward(self, input):
        self.input = input
        self.output = self.activation(self.input)

        return self.output

    def backward(self, output_error, learning_rate):
        return self.activation_derivative(self.input) * output_error


class NeuralNetwork():
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_derivative = None

    def choose_error(self, loss, loss_derivative):
        self.loss = loss
        self.loss_derivative = loss_derivative

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, input_data):
        inputlen = len(input_data)
        result = []
        for i in range(inputlen):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)
        return result


def save_model(model, error, filename=f"vcomplete_model"):
    dirname = "/trainedmodels/"
    filepath = os.path.dirname(__file__) + dirname + filename
    with open(filepath + str(error) + ".pkl", "wb") as file:
        pickle.dump(model, file)


def train(model, x_train, y_train, epochs, learning_rate):
    leninput = len(x_train)
    for i in range(epochs):
        err = 0
        for j in range(leninput):
            output = x_train[j]
            for layer in model.layers:
                output = layer.forward(output)
            err += model.loss(y_train[j], output)
            error = model.loss_derivative(y_train[j], output)
            for layer in reversed(model.layers):
                error = layer.backward(error, learning_rate)
        err /= leninput
        # print('epoch %d/%d   error=%f' % (i+1, epochs, err))

        # save the model if it is a good model < 5 MSE
        if err <= 5:
            save_model(model, err)
