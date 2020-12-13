# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 13:07:10 2020

Nneural network that can classify handwritten numbers

1. Build a neural network that classifies images.
2. Train this neural network.
3. And, finally, evaluate the accuracy of the model.

TUTORIAL - Coding
https://www.tensorflow.org/tutorials/quickstart/beginner

TUTORIAL - Basic understanding
https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi

GITHUB
https://github.com/Paulilein/MNIST-NeuralNetworkNumberClassification

@author: pauli
"""

from tensorflow import keras
import tensorflow as tf

mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# x = image = grid of pixel intensity
# y = vector of length 10 where each position indicates a digit range 0 to 9. 
#      0 1 2 3 4 5 6 7 8 9
# y = [0 0 0 1 0 0 0 0 0 0] corresponds to digit 3

x_train, x_test = x_train / 255.0, x_test / 255.0
# Intensity ranges between 0 and 255 (256 = 2^8 = 1 byte = 8 bit)
# Division of 255 to get a range between 0 and 1

model = tf.keras.models.Sequential([
    keras.layers.Flatten(input_shape = (28, 28)),
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10)
    ])

# model is my neural network
# Sequential means that the network is made up of sequential layers
# Flatten transforms the 28x28 matrix to a 784 (28x28) vector, this corresponds to the first layer
# Dense means that between the two layers, each and every neuron is connected 
# to every neuron of the previous layer. Next layer has 128 neurons
# Dropout solves "overfitting". If free to fit a solution, perfect for the datapoints
# given, the solution will be perfect for solving this set of data but useless for any other
# dataset. Dropout(0.2) means that each new input 20% of the signals between the two layers
# are randomly set to zero. This will then result in a more robust solution where
# the neurons are less indipendent.
# Last Dense creates the output with 10 alternatives.