# Dot product - For loop vs cosine method vs dot function

import numpy as np

a = np.array([1,2])

b = np.array([2,1])

# Zip two vectors
zipped = zip(a,b)
print zipped

# Finding the dot product using loop
dot = 0
for i,j in zip(a,b):
	dot += i*j
print dot

print "Element-wise multiplication"
print a*b

# Methods to perform dot product in NumPy
print np.sum(a*b)

print (a*b).sum()

print np.dot(a,b)

print a.dot(b)

print b.dot(a)

# Finding magnitude of a - Method 1
amag = np.sqrt((a*a).sum())

# Finding magnitude of a - Method 2
amag = np.linalg.norm(a)
bmag = np.linalg.norm(b)

print "|a|"
print amag
print "|b|"
print bmag

# Calculate the cosine angle of the dot product
cosangle = a.dot(b) / ((amag) * (bmag))

print "cos(angle)"
print cosangle

# Calculate the inverse of cosine angle
angle = np.arccos(cosangle)

print "angle"
print angle