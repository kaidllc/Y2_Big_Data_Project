# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 12:15:34 2021

This is a simple neural network following the instructions from:
    https://medium.com/better-programming/how-to-create-a-simple-neural-network-in-python-dbf17f729fe6
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Here we define the neural network

class NeuralNetwork():
    def __init__(self,):
        self.inputSize = 3
        self.outputSize = 1
        self.hiddenSize = 3
        
        self.W1 = np.random.rand(self.inputSize, self.hiddenSize)
        self.W2 = np.random.rand(self.hiddenSize, self.outputSize)
        
        self.error_list = []
        self.limit = 0.5
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0
        
    def forward(self, X):
        self.z = np.matmul(X, self.W1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.matmul(self.z2, self.W2)
        o = self.sigmoid(self.z3)
        return o
    
    def sigmoid(self,s):
        return 1/(1 + np.exp(-s))
    
    def sigmoidPrime(self,s):
        return s*(1-s)
    
    def backward(self,X,y,o):
        self.o_error = y - o
        self.o_delta = self.o_error * self.sigmoidPrime(o)
        self.z2_error = np.matmul(self.o_delta,
                                  np.matrix.transpose(self.W2))
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)
        self.W1 += np.matmul(np.matrix.transpose(X), self.z2_delta)
        self.W2 += np.matmul(np.matrix.transpose(self.z2),
                             self.o_delta)


"""
To make sure that the model is evaluated based on how good 
it is to predict new data points, and not how well it is 
modeled to the current ones, it is common to split the 
datasets into one training set and one test set (and 
sometimes a validation set).
"""

input_train = np.array([[0, 1, 0], [0, 1, 1], [0, 0, 0], 
                        [10, 0, 0], [10, 1, 1], [10, 0, 1]])

output_train = np.array([[0], [0], [0], [1], [1], [1]])

input_pred = np.array([1, 1, 0])

input_test = np.array([[1, 1, 1], [10, 0, 1], [0, 1, 10], 
                       [10, 1, 10], [0, 0, 0], [0, 1, 1]])

output_test = np.array([[0], [1], [0], [1], [0], [0]])

"""
This MinMaxScaler scales and translates each feature individually 
such that it is in the given range on the training set, e.g. 
between zero and one. 
"""

scaler = MinMaxScaler()
input_train_scaled = scaler.fit_transform(input_train)
output_train_scaled = scaler.fit_transform(output_train)
input_test_scaled = scaler.fit_transform(input_test)
output_test_scaled = scaler.fit_transform(output_test)


