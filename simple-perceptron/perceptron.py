# Organizing imports
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn import datasets
import imutils
import cv2
import numpy as np

# Loading the IRIS dataset provided by scikit-learn
dataset = datasets.load_iris()

# Splitting the training and testing data using test_size parameter as 0.25
# Training data --> 75%
# Testing data  --> 25%
(trainData, testData, trainLabels, testLabels) = train_test_split(dataset.data, dataset.target, test_size=0.25, random_state=42)

# Training the perceptron model
print("[STATUS] Training the perceptron model..")
perceptron_model = Perceptron(n_iter=10, eta0=1.0, random_state=84)
perceptron_model.fit(trainData, trainLabels)

# Making predictions on the perceptron model
print("[STATUS] Making predictions..")
predictions = perceptron_model.predict(testData)

# Display the evaluation made by the perceptron model
print(classification_report(predictions,testLabels, target_names = dataset.target_names))