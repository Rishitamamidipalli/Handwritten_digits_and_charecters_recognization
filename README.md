# Hand written character and digit recognition using CNN
This project implements a Convolutional Neural Network (CNN) for recognizing handwritten characters and digits.
# Table of Contents
- [Dataset](#dataset)

### Dataset
It combines the MNIST dataset for digit recognition and a custom dataset for character recognition from kaggle.
you can download character dataset from:https://www.kaggle.com/datasets?search=a-z+characters+dataset

MNIST data set contains 70000 didgits data of 28*28 pixels format which can be loaded using mnist.load_data()
and character datadest contains 372450 characters data in csv file in which each image is flattened into row of 784(28*28) pixels

# Technologies Used:
1. Convolutional Neural Network (CNN)
2. Libraries: Pandas,Numpy,sklearn,Tensorflow,CV2,Keras,streamlit
2. Streamlit: Streamlit is an open-source Python library for creating interactive web apps for data science and machine learning projects easily

# How to execute code
1. First download the dataset from kaggle and add the path to the Pre_processing.py and then run that file you will get test_x.npy,train_x.npy,test_y.npy,train_y.npy which are numpy files whereour preprocessed data is stored.
2. Now run the Training.py file the model will be trained using the preprocessed numpy files and model will be saved.
3. So atlast we need to run Website.py in which a local webpage is opened in chrome which allows users to input an image and model will predict the out put
4. If you encounter any errors in the Webpage please reload the page again 

# DEMO

https://github.com/Rishitamamidipalli/Handwritten_digits_and_charecters_recognization/assets/123208162/7f6bd419-c34c-442e-b9ef-76d65f3503c5

# Model Architechure
CNN model consists of total 5 layers in which 2 are convolution layers and other two are fully connected dense layer

Input Layer: The input layer defines the shape of the input data. In this model, the input shape is (28, 28, 1), indicating grayscale images of 28x28 pixels.

Convolutional Layers: Two convolutional layers are employed, each with 32 filters of size 5x5. These layers learn to extract features from the input images through convolutions.

Batch Normalization Layers: Batch normalization is applied after each convolutional layer to standardize the activations, stabilizing and accelerating the training process.

MaxPooling Layer: Following the second convolutional layer is a max-pooling layer with a pooling window of size 5x5. MaxPooling reduces spatial dimensions, aiding computational efficiency and preventing overfitting.

Dropout Layer: Dropout regularization is applied to reduce overfitting. A dropout rate of 0.25 is employed to randomly deactivate 25% of neurons during training.

Flatten Layer: The output from the last convolutional layer is flattened into a one-dimensional vector to prepare it for input into the fully connected layers.

Dense Layers: Two dense (fully connected) layers are included. The first dense layer has 256 units with ReLU activation, while the second layer has 36 units with softmax activation, representing the number of output classes (digits 0-9 and characters A-Z)

# Results :
Training accuracy is about 98.5 for 5 epochs

Testing accuracy:98 percent
