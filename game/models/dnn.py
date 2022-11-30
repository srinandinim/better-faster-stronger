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


print(vectorize_coordinate(3))
print(vectorize_state((1,2,3)))



