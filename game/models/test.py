import numpy as np

A = np.array([1.25488586, -2.74511414, -3.74511414, 1.25488586])

# Use the reshape() method to convert A to a 2-dimensional array with 1 column and 4 rows
B = A.reshape(4, 1)

# Print the resulting array
print(B)