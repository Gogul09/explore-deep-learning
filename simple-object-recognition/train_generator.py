import numpy as np
import pandas as pd
import os
import cv2
import glob

# seed for reproducing same result
seed = 9
np.random.seed(seed)

# get the training image path
train_path = "dataset/train"

# get the training labels
train_labels = os.listdir(train_path)
train_labels.sort()

# resize the images to a fixed dimension
fixed_row = 28
fixed_col = 28
total_images = 51
num_pixels = fixed_row * fixed_col

# matrix to store all the training images
flattened_array = np.zeros(shape=(total_images, num_pixels))

# vector to store all the labels
list_labels = np.zeros(shape=(total_images, 1))

i = 0
j = 0
# loop through all the training images 
for training_name in train_labels:
	# get the path to current image class directory
	dir = os.path.join(train_path, training_name)
	# loop through all images in a single class
	for file in glob.glob(dir + "/*.jpg"):
		# read the image 
		img = cv2.imread(file)

		# convert to grayscale
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		# resize and normalize it
		gray = cv2.resize(gray, (fixed_row, fixed_col), interpolation=cv2.INTER_AREA)
		gray = gray.astype("float32")
		gray = gray/255.0

		# flatten the image
		flat = gray.flatten()

		# store the flattened image in a larger matrix
		flattened_array[i] = flat

		# store the associated image label
		list_labels[i] = j
		i += 1
	j += 1

# concatenate the training images and labels
concatenated = np.concatenate((list_labels, flattened_array), axis=1)

# save it to a csv file
np.savetxt("train.csv", concatenated, delimiter=",")
print "[STATUS] Training data saved!"