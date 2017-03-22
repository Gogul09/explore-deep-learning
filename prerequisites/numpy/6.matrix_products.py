# Matrix multiplication
# Inner dimensions should match

# . -> Matrix multiplication
# * -> Element-wise multiplication

import numpy as np

# Create two matrices
A = np.array([[1,2],[2,1]])
B = np.array([[3,1],[1,3]])

# Multiply two matrices
print A.dot(B)
