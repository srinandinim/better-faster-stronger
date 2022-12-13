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
        self.output_size = output_size
        self.input_size = input_size
        self.inputs = []
    
    def forward(self, input):
        self.inputs.append(input)
        self.input = input
        self.output = np.dot(self.input, self.w) + self.b
        return self.output

    def backward(self, output_error, lr):
        # output_error is a vector of errors
        # input_error is a pass back vector of errors to give to the previous layer
        input_error = np.dot(output_error, self.w.T)
        weights_error = np.asarray([], dtype="float32")
        for i in range(0, len(output_error)):
            weights_error = np.append(weights_error, np.dot(self.inputs[i].T, output_error[i]))
        # weights_error = np.dot(self.input.T, output_error.T)
        # print("Gradient: ", weights_error)
        for i in range(0, len(output_error)):
            self.w -= lr * weights_error[i]
            self.b -= lr * output_error[i]

        return input_error

class NonLinearity(Layer):
    def __init__(self, activation, activation_derivative):
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.inputs = []

    def forward(self, input):
        self.inputs.append(input)
        self.input = input
        self.output = self.activation(self.input)
        return self.output

    def backward(self, output_error, learning_rate):
        return np.asarray([self.activation_derivative(self.inputs[i]) * output_error[i] for i in range(0, len(self.inputs))], dtype="float32")

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

    def compute_output(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def back_propagate(self, expected, output, learning_rate):
        error = self.loss_derivative(expected, output)
        for layer in reversed(self.layers):
            error = layer.backward(error, learning_rate)
        self.clear_layer_caches()

    def clear_layer_caches(self):
        for layer in self.layers:
            layer.inputs = []

def save_model(model, error, testerror = 0, filename=f"vpartial_model"):
    dirname = "/trainedmodels/"
    filepath = os.path.dirname(__file__) + dirname + filename
    with open(filepath + str(error) + "_" + str(testerror) + ".pkl", "wb") as file:
        pickle.dump(model, file)

def train(model, x_train, y_train, epochs, learning_rate):
    leninput = len(x_train)
    for i in range(epochs):
        err = 0
        for j in range(leninput):
            #print(j)
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
    leninput = len(x_train)
    for i in range(epochs):
        err = 0
        for j in range(leninput):
            #print(j)
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

        print('epoch %d/%d   train_error=%f   test_error=%f' % (i+1, epochs, err, mse_testerror))
    
        # save the model if it is a good model < 5 MSE
        if err <= 5 and mse_testerror <= 5:
            save_model(model, err, mse_testerror)
