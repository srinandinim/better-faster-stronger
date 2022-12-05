import time 
import pickle
import numpy as np 

# FUNCTIONS TO RETREIVE TRAINING AND TEST DATA 

def get_training_data(data=np.loadtxt("trainingdata/vcomplete_trainingdata.csv", delimiter=","), start_idx=0, end_idx=125000):
    """
    retrieves the start:end datapoints for the targets Y and input features X
    """
    # loads the CSV file from numpy into memory
    Y, X = data[start_idx:end_idx,0], data[start_idx:end_idx,1:]
    print(Y.shape, X.shape)

    # make all negative infinity values predict to -50
    Y[Y == -np.inf] = -50

    # reshape the data so that we can input the work onto neural net
    X, Y = np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)
    X, Y = X.reshape((X.shape[0], 1, X.shape[1])), Y.reshape((Y.shape[0], 1))

    # returns the 
    return Y, X

# FUNCTIONS AND DERIVATIVES FOR ACTIVATION/LOSS

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1 - np.tanh(x)**2

def ReLU(x):
    return max(0,x)

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

# TRAIN THE NEURAL NET, DESERIALIZE IT, AND SERIALIZE IT
  
def save_model(model, error, filename=f"trainedmodels/vcomplete_model"):
    with open(filename + str(error) + ".pkl", "wb") as file:
        pickle.dump(model, file)
        print("model successfully serialized")

def load_model(filename="trainedmodels/vcomplete_model.pkl"):
    with open(filename, "rb") as file:
        model = pickle.load(file)
        print("model successfully deserialized")
    return model 

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
        print('epoch %d/%d   error=%f' % (i+1, epochs, err))

        # save the model if it is a good model < 5 MSE
        if err <= 5: 
            save_model(model, err)

# TOY EXAMPLE TO TEST FUNCTIONALITY OF NEURAL NET
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
dnn_v_complete = NeuralNetwork()
dnn_v_complete.add(DenseLinear(2, 1000))
dnn_v_complete.add(NonLinearity(tanh, tanh_prime))
dnn_v_complete.add(DenseLinear(1000, 1000))
dnn_v_complete.add(NonLinearity(tanh, tanh_prime))
dnn_v_complete.add(DenseLinear(1000, 1))


# train
#dnn_v_complete.choose_error(mse, mse_prime)
#start = time.time()
#train(dnn_v_complete, x_train, y_train, epochs=1000, learning_rate=0.001)
#print(f"training took {time.time()-start} seconds")

# test
#ut = dnn_v_complete.predict(x_train)
#print(out)

#model = load_model("trainedmodels/vcomplete_model0.022229802029378.pkl")
#print(model.predict(x_train))