# Vector - 1D NumPy array
# Matrix - 2D NumPy array

import numpy as np

# Create a NumPy matrix
M = np.array([ [1,2], [3,4] ])

# Create a list of list - similar to matrix
L = [ [1,2], [3,4] ]

print L[0]
print L[0][0]

print M[0][0]
print M[0,0]

# Create a NumPy matrix
M2 = np.matrix([ [1,2],[3,4] ])
print M2

# Changing a matrix to array
A = np.array(M2)
print A

# Transpose of a matrix/array
print A.T