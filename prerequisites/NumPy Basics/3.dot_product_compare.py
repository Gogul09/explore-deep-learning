# Comparing Dot product using two methods

import numpy as np
from datetime import datetime

# Generate some random 1D-array
a = np.random.randn(100)
b = np.random.randn(100)
T = 100000

# Definition for a slow dot product
def slow_dot_product(a,b):
	result = 0
	for e, f in zip(a,b):
		result += e*f
	return result

# Getting the time for slow dot product
t0 = datetime.now()
for t in xrange(T):
	slow_dot_product(a,b)
dt1 = datetime.now() - t0

# Getting the time for numpy dot product
t0 = datetime.now()
for t in xrange(T):
	a.dot(b)
dt2 = datetime.now() - t0

# Finding the difference
print " dt1 / dt2: ", dt1.total_seconds() / dt2.total_seconds()