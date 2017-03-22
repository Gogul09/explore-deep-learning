import numpy as np

# A simple list
L = [1,2,3]
print "A simple list"
print L

# A simple NumPy array
A = np.array([1,2,3])
print "A simple NumPy array"
print A

# Loop through each element in list
print "Loop through list"
for e in L:
     print e

# Loop through each element in array
print "Loop through array"
for e in A:
     print e

# Append an element to list
print "Append 4 to list"
L.append(4)
print L

# Appending won't work on array
# A.append(4)

# Concatenating an element with list
print "Concatenate 5 with list"
L = L + [5]
print L

# Concatenation won't work with array
# A = A + [4,5]

# Vector addition in list
L2 = []
print "Vector addition - list"
for e in L:
     L2.append(e + e)
print L2

# Vector addition in array
print "Vector addition - array"
print A + A

# Scalar multiplication in array
print "Scalar multiplication - array"
print 2*A

# Doubling the list
print "Doubling - list"
print 2*L

# Squaring each element in list
L2 = []
print "Square of each element - list"
for e in L:
	L2.append(e*e)
print L2

# Squaring each element in array
print "Square of each element - array"
print A**2

# Square root of each element in array
print "Square root of each element - array"
print np.sqrt(A)


# Log of each element in array
print "Log of each element - array"
print np.log(A)

# Exponent of each element in array
print "Exponent of each element - array"
print np.exp(A)