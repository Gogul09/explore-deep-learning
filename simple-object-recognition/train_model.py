import pandas as pd
import numpy as np
import cv2
import os
import glob
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils

# Convolutional Neural Network
# [CONV2D]->[POOL2D]->[DROPOUT]->[FLATTEN]->[DENSE]
def create_model():
	# create model
	model = Sequential()
	model.add(Convolution2D(num_pixels, 5, 5, input_shape=(28, 28, 1), activation= "relu"))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation= "relu"))
	model.add(Dense(num_classes, activation= "softmax"))
	# compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# seed for reproducing results
seed = 9
np.random.seed(seed)

# read the dataset
train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')

# extract image pixels and labels from training dataset
images = train.iloc[:,1:].values
labels = train.iloc[:,0].values

# extract image pixels from testing dataset
test_images = test.iloc[:,0:].values

# reshape the images for sending it to the network
images = images.reshape(images.shape[0], 28, 28, 1).astype('float32')
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')
num_pixels = images.shape[1]

# encode the training labels as integers
encoded_labels = np_utils.to_categorical(labels)
num_classes = encoded_labels.shape[1]

# create the model
model = create_model()

# fit the model
model.fit(images, encoded_labels, nb_epoch=10, batch_size=10, verbose=2)

# evaluate the model
scores = model.predict(test_images)
scores = scores.argmax(1)

# save result
np.savetxt('result.csv', np.c_[range(1, len(test)+1),scores], delimiter=',', header='ImageID, Label', comments="", fmt="%d")
print "[STATUS] Prediction saved!"

# get access to test images
test_path = "dataset/test"

# get access to the labels
train_path = "dataset/train"
train_labels = os.listdir(train_path)
train_labels.sort()
images_path = os.listdir(test_path)

# loop over all the test images
for i in range(len(images_path)):
	# get the path to each individual image
	file = os.path.join(test_path, images_path[i])

	# read the image
	img = cv2.imread(file)

	# resize the image for better visualization in result
	show = cv2.resize(img, (512,512), interpolation=cv2.INTER_AREA)

	# convert to grayscale and resize it
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gray = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

	# reshape the test image to send it to the network
	test_image = gray.reshape(1, 28, 28, 1).astype('float32')

	# normalize the test image
	test_image = test_image/255.0

	# predict the test image
	prediction = model.predict(test_image)

	# take only the label output
	prediction = prediction.argmax(1)

	# label integer to string
	predicted_label = train_labels[prediction]

	print "I think it is: " + predicted_label 

	# visualize the result
	cv2.putText(show, predicted_label, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255),2)
	cv2.imshow("Image", show)
	cv2.waitKey(0)

cv2.destroyAllWindows()