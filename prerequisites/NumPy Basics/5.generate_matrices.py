# Generate arrays of data

import numpy as np

# Standard method
A = np.array([1,2,3])
print A

# Create a vector of all zeros
Z = np.zeros(10)
print Z

# Create a matrix of all zeros
X = np.zeros((10,10))
print X

# Creating a matrix of all ones
O = np.ones((10,10))
print O

# Creating a random matrix - Using tuple
M = np.random.random((2,2))
print M

# Creating a random matrix - No tuple for randn
N = np.random.rand(2,2)
print N

# Finding Mean
print "Mean"
print M.mean()

# Finding Variance
print "Variance"
print M.var()