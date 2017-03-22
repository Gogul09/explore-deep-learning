# Problem: The admission fee at a small fair is $1.50 for children and $4.00 for adults. On a certain day, 2200 people enter the fair and $5050 is collected. How many children and how many adults attended?

# Solution:
# x1 = number of children
# x2 = number of adults
# x1 + x2 = 2200
# 1.5x1 + 4x2 = 5050

import numpy as np

A = np.array([[1,1],[1.5,4]])
print A

B = np.array([2200,5050])
print B

# Traditional way
Ainv = np.linalg.inv(A)
print Ainv

X = Ainv.dot(B)
print X

# More easier way
Z = np.linalg.solve(A,B)
print Z