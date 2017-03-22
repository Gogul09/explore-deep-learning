import numpy as np
import pandas as pd
import os
import cv2
import glob

# seed for reproducing same result
seed = 9
np.random.seed(seed)

# get the training image path
test_path = "dataset/test"
image_paths = os.listdir(test_path)

# resize the images to a fixed dimension
fixed_row = 28
fixed_col = 28
total_images = len(image_paths)
num_pixels = fixed_row * fixed_col

# matrix to store all the testing images
flattened_array = np.zeros(shape=(total_images, num_pixels))

# loop through the test images
for i in range(len(image_paths)):
	# get the path to each individual image
	file = os.path.join(test_path, image_paths[i])

	# read the image
	img = cv2.imread(file)

	# convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# resize the image and normalize
	gray = cv2.resize(gray, (fixed_row, fixed_col), interpolation=cv2.INTER_AREA)
	gray = gray.astype("float32")
	gray = gray/255.0

	# flatten the image
	flat = gray.flatten()

	# store it in a larger matrix
	flattened_array[i] = flat
	i += 1

# save test features to a csv file
np.savetxt("test.csv", flattened_array, delimiter=",")
print "[STATUS] Testing data saved!"