import random 
from node import Node 

class Neuron():
    def __init__(self, n_inputs):
        self.w = [Node(random.uniform(-1,1)) for _ in range(n_inputs)]
        self.b = Node(random.uniform(-1,1))

    def __call__(self, x, activated=True):
        unactivated = self.b
        for wi, xi in zip(self.w, x):
            unactivated += wi * xi

        if activated: return unactivated.sigmoid()
        else: return unactivated 
        
    def parameters(self):
        return self.w + [self.b]

class Layer():
    def __init__(self, n_inputs, n_output):
        self.neurons = [Neuron(n_inputs) for _ in range(n_output)]
    
    def __call__(self, x, activated=True):
        outputs = [n(x, activated) for n in self.neurons]
        return outputs[0] if len(outputs)==1 else outputs
    
    def parameters(self):
        p = []
        for neuron in self.neurons:
            ps = neuron.parameters()
            p.extend(ps)
        return p

class DNN():
    def __init__(self, n_inputs, n_outputs):
        size = [n_inputs] + n_outputs
        self.layers = [Layer(size[i], size[i+1]) for i in range(len(n_outputs))]
    
    def __call__(self, x):
        for i, layer in enumerate(self.layers): 

            # the last layer is linear 
            if i == len(self.layers)-1:
                x = layer(x, activated=False)

            # all other layers are non-linear
            else: 
                x = layer(x, activated=True)

        return x 
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

def compute_mse_loss():
    loss = []
    for y, y_hat in zip(ys, ypred):
        y = Node(y)
        loss.append((y + (-y_hat) )**2)
    return loss 

def compute_total_loss(loss_list):
    total_loss = Node(0)
    for loss in loss_list:
        total_loss += loss 
    return total_loss

# arbitrary xi
x = [2.0, 3.0, -1.0]

# initialized DNN 
dnn = DNN(3, [4,4,1])

# initial output for arbitrary DNN
print("initial output for arbitrary DNN")
print(dnn(x))

# how many network parameters to optimize
print("how many network parameters to optimize")
print(len(dnn.parameters()))

# dataset for training
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]
ys = [2.0, -2.0, -2.0, 2.0]

# predict outputs for every item in training data
print("predict outputs for every item in training data")
ypred=[dnn(x) for x in xs]
print(ypred)

# compute error between true and predicted targets
print("# compute error between true and predicted targets")
loss_list = compute_mse_loss()
print(loss_list)
loss = compute_total_loss(loss_list)
print(loss)

# see the data and the gradients before
print("see the data and the gradients before backprop")
print(dnn.layers[0].neurons[0].w[0].data)
print(dnn.layers[0].neurons[0].w[0].grad)

# compute backprop on the layers
loss.backward()

# update the parameters of the network based on gradient descent
for p in dnn.parameters():
    p.data += -0.01 * p.grad

# see the data and the gradients
print("see the data and the gradients after backprop")
print(dnn.layers[0].neurons[0].w[0].data)
print(dnn.layers[0].neurons[0].w[0].grad)

# predict outputs for every item in training data
print("predict outputs for every item in training data")
ypred=[dnn(x) for x in xs]
print(ypred)

# compute error between true and predicted targets
print("# compute error between true and predicted targets")
loss_list = compute_mse_loss()
print(loss_list)
loss = compute_total_loss(loss_list)
print(loss)

