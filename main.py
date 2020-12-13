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
Using HTTPS:
https://github.com/Paulilein/MNIST-NeuralNetworkNumberClassification
Using SSH:
# git@github.com:Paulilein/MNIST-NeuralNetworkNumberClassification.git

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
# 'relu' is an activation function. The function is first zero and than linear

# Dropout solves "overfitting". If free to fit a solution, perfect for the datapoints
# given, the solution will be perfect for solving this set of data but useless for any other
# dataset. Dropout(0.2) means that each new input 20% of the signals between the two layers
# are randomly set to zero. This will then result in a more robust solution where
# the neurons are less indipendent.

# Last Dense creates the output with 10 alternatives.

predictions = model(x_train[:1]).numpy() # translate input to numpy array
print(predictions)

tf.nn.softmax(predictions).numpy()
# softmax is another function. It takes every singal from the layers and divides 
# it by the sum of all signals -> %. The highest persentage is the result the 
# neural network then guesses

loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits = True)
# The loss function calculates the error for the whole neural network. 
# If zero the model is sure of the correct class
# Untrainer, this model is as good as random. The result should then be around 2.3
# -tf.log(1/10) ~= 2.3

loss = loss_fn(y_train[:1], predictions).numpy()
print(loss)

model.compile(optimizer = 'adam',
              loss = loss_fn,
              metrics = ['accuracy'])
# To start training the neural network an optimizer has to be set. This gives
# the gradient of the error and shows the direction the correct value

# loss defines which lossfunction we use. We want to minimize the lossfunction
# using the optimizer selected

# metrics is what type of value we want information about during training. 
# 'accuracy' will show the percentage of images that the neural network
# cathegorizes correctly

model.fit(x_train, y_train, epochs = 5)
# usually epochs should be around 50 - 200. Its the number of iterations

model.evaluate(x_test, y_test, verbose = 2)
# The Model.evaluate method checks the models performance, usually on 
# a "Validation-set" or "Test-set".

#%% This is interesting when you use the network, not when you train it.
# when you train it you want it to be able to generate very large numbers. 
# Since those would then generate a big improvement which otherwise would be lost.
probability_model = keras.Sequential([
    model,
    keras.layers.Softmax()
    ])

probability_model(x_test[:5])