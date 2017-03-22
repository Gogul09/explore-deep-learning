# Hand-written Digits Recognition using CNN

A simple Convolutional Neural Network built with Keras and TensorFlow backend for recognizing hand-written digits.
To test this project, you need the kaggle dataset for digits recognition. You need both "train.csv" and "test.csv"

### Network Architecture
[CONV2D]->[POOL2D]->[DROPOUT]->[FLATTEN]->[DENSE]->[DENSE]

### Activation functions
- Rectifier (relu)
- Softmax

### Libraries used
- Keras
- Pandas
- NumPy

### Parameters
- Optimizer: Adam
- Loss function: Categorical cross-entropy
- No.of.Epochs: 10
- Batch size: 10

### Accuracy = <strong> 99.71% </strong>

### Result
![alt tag](https://github.com/Gogul09/Exploring-CV-DL-ML/blob/master/Deep_Learning/digits-recognizer/output.png)
