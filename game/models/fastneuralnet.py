import time 
import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2

def identity(x):
    return x

def identity_prime(x):
    return 1

def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size

class Layer:
    def __init__(self):
        self.input, self.output = None, None 

    def forward_propagation(self, input):
        """IMPLEMENT THIS IN OTHER CLASSES"""
        pass 

    def backward_propagation(self, output_error, learning_rate):
        """IMPLEMENT THIS IN OTHER CLASSES"""
        pass 

class DenseLinear(Layer):

    def __init__(self, input_size, output_size):
        self.w = np.random.rand(input_size, output_size) - 0.5
        self.b = np.random.rand(1, output_size) - 0.5

    def forward(self, input):
        self.input = input
        self.output = np.dot(self.input, self.w) + self.b
        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
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

    def backward(self, output_error, lr = None):
        return self.activation_derivative(self.input) * output_error

class Network:
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

    def train(self, x_train, y_train, epochs, learning_rate):
        leninput = len(x_train)

        for i in range(epochs):
            err = 0
            for j in range(leninput):
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward(output)

                # compute loss (for display purpose only)
                err += self.loss(y_train[j], output)

                # backward propagation
                error = self.loss_derivative(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward(error, learning_rate)

            # calculate average error on all samples
            err /= leninput
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))

"""


# training data
x_train = np.array(
    [
        [ [0,0] ], 
        [ [0,1] ], 
        [ [1,0] ], 
        [ [1,1] ]
    ]
)

y_train = np.array(
    [
        [[-9]], [[9]], [[8]], [[-27]]
    ]
)

print(x_train)
print(x_train.shape)
print(y_train)
print(y_train.shape)


# network
dnn_v_complete = Network()
dnn_v_complete.add(DenseLinear(2, 1000))
dnn_v_complete.add(NonLinearity(tanh, tanh_prime))
dnn_v_complete.add(DenseLinear(1000, 1000))
dnn_v_complete.add(NonLinearity(tanh, tanh_prime))
dnn_v_complete.add(DenseLinear(1000, 1))


# train
dnn_v_complete.choose_error(mse, mse_prime)
start = time.time()
dnn_v_complete.train(x_train, y_train, epochs=1000, learning_rate=0.001)
print(f"training took {time.time()-start} seconds")

# test
out = dnn_v_complete.predict(x_train)
print(out)

"""
