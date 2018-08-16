# Simple Object Recognition using Keras

This project demonstrates -
* How to load training (having separate folder for each label) and testing images from disk
* How to convert training and testing images to CNN-ready format (i.e flatten the image matrix)
* How to save the flattened images to .csv files
* How to construct a simple CNN model using Keras
* How to perform prediction on images stored in disk

### Libraries used
* NumPy
* Keras
* Pandas

### Accuracy = 100% 

Accuracy is maximum because it is just a smaller dataset. This project aims to provide a blueprint to start more complex models using dataset that is residing in local disk. Keras provides methods to fetch standard datasets like MNIST, CIFAR-10 etc. But I couldn't find anything that tells how to prepare a dataset ready for CNN from disk. So, I planned to implement it myself. Feel free to use this code!
