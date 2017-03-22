import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils

# create the model
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

# seed for reproducing same result
seed = 9
np.random.seed(seed)

# load the dataset
train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')

# extract image pixels
images = train.iloc[:,1:].values
test_images = test.iloc[:,0:].values

# reshape the images
images = images.reshape(images.shape[0], 28, 28, 1).astype('float32')
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')

# normalize images
images = images / 255.0
test_images = test_images / 255.0
num_pixels = images.shape[1]

# extract labels
labels = train.iloc[:,0].values

# encode output labels as integers
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