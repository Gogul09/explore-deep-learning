import numpy as np

# Create a matrix
A = np.array([[1,2],[3,4]])
print "A"
print A

# Find inverse
Ainv = np.linalg.inv(A)
print "A inverse"
print Ainv

# Find whether Identity matrix arrives
print "Identity Matrix = Ainv * A"
print Ainv.dot(A)
print A.dot(Ainv)

# Find determinant
D = np.linalg.det(A)
print "Determinant of A"
print D

# Find diagonal --> returns vector
# Passing 2D array returns a 1D array
diag = np.diag(A)
print "Diagonal of A - 1D array"
print diag

# Passing a 1D array returns a 2D array
print "Diagonal of A - 2D array"
print np.diag([1,2])

I = np.array([1,2])
J = np.array([3,4])
print "I"
print I
print "J"
print J

# Find outer product of two vectors
print "Outer product of I and J"
print np.outer(I,J)

# Find inner product of two vectors
print "Inner product of I and J"
print np.inner(I,J)
print I.dot(J)

# Find trace
print "Trace of A"
print np.trace(A)
print np.diag(A).sum()

# Eigen values and Eigen vectors
# Each sample takes the row 
# Each column takes the feature
# Sample - 100; Feature - 3
X = np.random.randn(100,3)

# Covariance of a matrix
# Remember to transpose it first!
cov = np.cov(X.T)
print cov

# np.eigh(C) --> For symmetric and Hermitian matrix
(eigenvalues, eigenvectors) = np.linalg.eigh(cov)
print eigenvalues
print eigenvectors