import random 
from node import Node 

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):
    def __init__(self, n_inputs):
        self.w = [Node(random.uniform(-1,1)) for _ in range(n_inputs)]
        self.b = Node(random.uniform(-1,1))

    def __call__(self, x, activated=True):
        unactivated = self.b
        for wi, xi in zip(self.w, x):
            unactivated += wi * xi
        if activated == False: 
            return unactivated
        else: 

            # this prevents overflow errors
            if unactivated.data > 2:
                unactivated.data = 2 
            elif unactivated.data < -2:
                unactivated.data = -2
            
            return unactivated.tanh()
        
    def parameters(self):
        return self.w + [self.b]

class Layer(Module):
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

class DNN(Module):
    def __init__(self, n_inputs, n_outputs):
        size = [n_inputs] + n_outputs
        self.layers = [Layer(size[i], size[i+1]) for i in range(len(n_outputs))]
    
    def __call__(self, x):
        for i, layer in enumerate(self.layers): 
            if i == (len(self.layers)-1):
                x = layer(x, activated=False)
            else: x = layer(x, activated=True)
        return x 
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

def compute_mse_loss_list(predictions, targets):
    loss = []
    for y, y_hat in zip(targets, predictions):
        y = Node(y)
        loss.append((y + (-y_hat) )**2)
    return loss 

def compute_total_loss(loss_list):
    total_loss = Node(0)
    for loss in loss_list:
        total_loss += loss 
    return total_loss


