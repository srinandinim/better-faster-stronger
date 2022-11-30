import numpy as np

def vectorize_coordinate(coordinate, length=50):
    """
    Takes a coordinate point and converts it to one hot vector. 

    Example:
    input: coordinate=3
    output: [0, 0, 1, 0, 0, 0, ..., 0, 0]
    """

    vector = []
    for i in range(length):
        if i == (coordinate-1): vector.append(1)
        else: vector.append(0)
    return vector 

def vectorize_state(state):
    """
    Takes a state and converts it to one hot vector matrix. 

    Example:
    input: state=(1,2,3)
    output: 
           [1, 0, 0, 0, 0, 0, ..., 0, 0]
           [0, 1, 0, 0, 0, 0, ..., 0, 0]
           [0, 0, 1, 0, 0, 0, ..., 0, 0]
    """
    matrix = [] 
    x,y,z = state 
    matrix.append(vectorize_coordinate(x))
    matrix.append(vectorize_coordinate(y))
    matrix.append(vectorize_coordinate(z))
    return matrix

class Value: 
    """stores matrices and their gradients"""

    def __init__(self, data, _children=(), _op=''):
        """
        data: numpy ndarray of values
        _children: list of Values that are children of the node
        _op: string representing operation that is done on the list of children
        """
        self.data = data 
        self._children = _children 
        self._op = _op 
        self.grad = np.zeros(data.shape)
    
    def __str__(self):
        """
        returns string representation of the Value
        """
        return f'Value: {self.data}'

    def backward(self):
        """
        performs backprop on the Value
        """
        if self._op == 'add':
            for child in self._children:
                child.grad += self.grad
                child.backward()
        elif self._op == 'sub':
            self._children[0].grad += self.grad
            self._children[1].grad -= self.grad
            self._children[0].backward()
            self._children[1].backward()
        elif self._op == 'dot':
            self._children[0].grad += np.dot(self.grad, self._children[1].data.T)
            self._children[1].grad += np.dot(self._children[0].data.T, self.grad)
            self._children[0].backward()
            self._children[1].backward()
        elif self._op == 'sigmoid':
            self._children[0].grad += self.grad * self.data * (1 - self.data)
            self._children[0].backward()
        elif self._op == 'relu':
            self._children[0].grad += self.grad * (self._children[0].data > 0)
            self._children[0].backward()
        elif self._op == 'mul':
            self._children[0].grad += self.grad * self._children[1].data
            self._children[1].grad += self.grad * self._children[0].data
            self._children[0].backward()
            self._children[1].backward()
        elif self._op == 'div':
            self._children[0].grad += self.grad / self._children[1].data
            self._children[1].grad += self.grad * self._children[0].data / self._children[1].data ** 2
            self._children[0].backward()
            self._children[1].backward()
        elif self._op == 'sum':
            self._children[0].grad += np.ones(self._children[0].data.shape) * self.grad
            self._children[0].backward()
        elif self._op == 'transpose':
            self._children[0].grad += np.sum(self.grad, axis=0)
            self._children[0].backward()
        elif self._op == 'pow':
            self._children[0].grad += self._children[1].data * self.grad * np.power(self._children[0].data, self._children[1].data - 1)
            self._children[1].grad += self.grad * np.power(self._children[0].data, self._children[1].data) * np.log(self._children[0].data)
            self._children[0].backward()
            self._children[1].backward()

    def set_grad(self, newGrad):
        """
        sets the gradient of the Value to newGrad
        :param newGrad: numpy ndarray of new gradient
        """
        self.grad = newGrad

    def __add__(self, other):
        """
        performs an addition operation
        :param other: other Value to add with
        :return: new Value that is sum of the two
        """
        return Value(self.data + other.data, [self, other], 'add')
        #return Value(self.data + other.data, _children=(self, other), _op='add')

    def __sub__(self, other):
        """
        performs a subtraction operation
        :param other: other Value to subtract with
        :return: new Value that is difference of the two
        """
        return Value(self.data - other.data, [self, other], 'sub')

    def __mul__(self, other):
        """
        performs a multiplication operation
        :param other: other Value to multiply with
        :return: new Value that is product of the two
        """
        return Value(self.data * other.data, [self, other], 'mul')

    def __truediv__(self, other):
        """
        performs a division operation
        :param other: other Value to divide with
        :return: new Value that is quotient of the two
        """
        return Value(self.data / other.data, [self, other], 'div')

    def dot(self, other):
        """
        performs a dot product
        :param other: other Value to dot product with
        :return: new Value that is dot product of the two
        """
        return Value(np.dot(self.data, other.data), [self, other], 'dot')

    def sigmoid(self):
        """
        performs a sigmoid operation on the Value
        :return: new Value that is sigmoid of the original
        """
        return Value(1/(1 + np.exp(-self.data)), [self], 'sigmoid')

    def relu(self):
        """
        performs a relu operation on the Value
        :return: new Value that is relu of the original
        """
        return Value(np.maximum(0, self.data), [self], 'relu')

    def sum(self, axis=None):
        """
        performs a sum operation on the Value
        :return: new Value that is sum of the original
        """
        return Value(np.sum(self.data, axis=axis), [self], 'sum')

    def transpose(self):
        """
        performs a transpose operation on the Value
        :return: new Value that is transpose of the original
        """
        return Value(self.data.T, [self], 'transpose')

    def pow(self, other):
        """
        performs a power operation
        :param other: other Value to power with
        :return: new Value that is power of the original
        """
        return Value(np.power(self.data, other.data), [self, other], 'pow')

def sigmoid(x):
    """ returns sigmoid of x """
    return 1/(1 + np.exp(-x))

def relu(x):
    """ returns relu of x """
    return np.maximum(0, x)

def square(x):
    """ returns square of x """
    return x**2

def add(x):
    """ returns sum of x """
    return np.sum(x)

x = Value(np.random.rand(3,50))
y = Value(np.random.rand(1))

#output size: 3 x 100
weights_1 = Value(np.random.rand(50,100))
biases_1 = Value(np.random.rand(100))

#output size: 3 x 100
weights_2 = Value(np.random.rand(100,100))
biases_2 = Value(np.random.rand(100))

#output size: 3 x 1
weights_3 = Value(np.random.rand(100,1))
biases_3 = Value(np.random.rand(1))

#output_1 = sigmoid(x.dot(weights_1) + biases_1)
#output_1 = relu(x.dot(weights_1).add(biases_1))
output_1 = x.dot(weights_1).add(biases_1).sigmoid()
output_2 = output_1.dot(weights_2).add(biases_2).sigmoid()
output_3 = output_2.dot(weights_3).add(biases_3)

# Mean squared error loss
loss = (output_3 - y).pow(2).sum()
loss.backward()

print(output_3.data)