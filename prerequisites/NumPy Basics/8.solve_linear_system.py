# Problem  : Ax = b
# Solution : Ainv A x = x = Ainv b

import numpy as np

A = np.array([[1,2],[3,4]])
B = np.array([1,2])

x = np.linalg.inv(A).dot(B)
print x

# Always use solve() to solve linear system
y = np.linalg.solve(A,B)
print y