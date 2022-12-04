import math

"""
Built with some reference to Andrej Karpathy's Micrograd YouTube Video.
Built with some reference to PyTorch Neural Network documentation. 
"""

class Node:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data 
        self._prev = set(_children)

        def _backward():
            return None 
        self._backward = lambda : _backward() 
        
        self._op = _op
        self.grad = 0 
        self.label = label
    
    def __add__(self, other):
        if not isinstance(other, Node): other=Node(other)
        out = Node(data=self.data+other.data, _children=(self,other), _op='+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out 
    
    def __mul__(self, other):
        if not isinstance(other, Node): other=Node(other)
        out = Node(data=self.data*other.data, _children=(self, other), _op='*')
        def _backward():
            self.grad += other.data * out.grad 
            other.grad += self.data * out.grad 
        out._backward = _backward
        return out 

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Node(self.data**other, (self,), _op="pow")

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out
    
    def sigmoid(self):
        x = self.data
        s = (1) / (1 + math.exp(-x))
        out = Node(data=s, _children=(self,), _op='sigmoid')
        def _backward():
            self.grad += s * (1-s) * out.grad
        out._backward = _backward
        return out
    
    def tanh(self):
        x = self.data 
        t = (math.exp(2*x)-1) / (math.exp(2*x) + 1)
        out = Node(data=t, _children=(self,))
        def _backward():
            self.grad += (1 - t**2) * out.grad 
        out._backward = _backward
        return out 

    def backward(self):

        # find order to do backprop
        ts = [] 
        visited = set() 
        def topsort(v):
            if v not in visited: 
                visited.add(v) 
                for child in v._prev: 
                    topsort(child)
                ts.append(v) 
        topsort(self) 
        
        # base case 
        self.grad = 1.0

        # backpropogation 
        for node in reversed(ts):
            node._backward() 

    def __neg__(self):
        return self * -1 
    
    def __sub__(self, other):
        return self + (-other)

    def __repr__(self):
        return f"Node(data={self.data}, grad={self.grad}, label={self.label})"
    
if __name__ == "__main__":

    """
    x1 = Node(2.0, label='x1')
    x2 = Node(0.0, label='x2')
    w1 = Node(-3.0, label='w1')
    w2 = Node(1.0, label='w2')
    b = Node(6.8813735870195432, label='b')
    x1w1 = x1*w1; x1w1.label = 'x1w1'
    x2w2 = x2*w2; x2w2.label = 'x2w2'
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1w1x2w2'
    n = x1w1x2w2+b; n.label = 'n'
    o = n.tanh(); o.label = 'o'
    o.backward()

    print(o)
    print(n)
    print(x1w1x2w2)
    print(x2w2)
    print(x1w1)
    print(b)
    print(w2)
    print(w1)
    print(x2)
    print(x1)
    """


