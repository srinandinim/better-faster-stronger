import math

"""
Built with some reference to Andrej Karpathy's Micrograd YouTube Video.
Built with some reference to PyTorch Neural Network documentation. 
"""

class Node:
    def __init__(self, data, _children=(), _op=''):
        self.data = data 
        self._prev = set(_children)
        self._backward = lambda : None 
        self._op = _op
        self.grad = 0 
    
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
            self.grad += other.data * out.data 
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
        return f"Node(data={self.data})"
    


if __name__ == "__main__":
    a = Node(2.0)
    b = Node(3.0)
    c = Node(10.0)
    d = a*b+c
    print(d._prev)
    print(d._op)