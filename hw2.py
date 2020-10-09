#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 00:30:59 2020

@author: Abhishek Jha

"""

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np



class MLP:
    
    def __init__(self, ip=13, hidden=[8], op=3):
        
        self.ip = ip
        self.hidden = hidden
        self.op = op
        
        layers = [self.ip] + hidden + [self.op]
        
        self.weights = []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i+1])
            self.weights.append(w)
            
        activations = []
        for i in range(len(layers)):
            a=np.zeros(layers[i])
            activations.append(a)
        self.activations = activations     
        
        derivatives = []
        for i in range(len(layers) - 1):
            d=np.zeros((layers[i], layers[i+1]))
            derivatives.append(d)
        self.derivatives = derivatives  
        
        
    def forward_propogation(self, inputs):
        
        activations = inputs
        self.activations[0] = inputs  
        
        for i, w in enumerate(self.weights):
            net_inputs = np.dot(activations, w)
            activations = self._sigmoid(net_inputs)
            self.activations[i+1] = activations            
        return activations
    
    def back_propagate(self, error, flag=False):
        
         for i in reversed(range(len(self.derivatives))):
             activations = self.activations[i+1]
             delta = error * self._sigmoid_derivative(activations)
             delta_reshaped = delta.reshape(delta.shape[0], -1).T
             current_activations = self.activations[i]
             current_activations_reshaped = current_activations.reshape(current_activations.shape[0], -1)
             self.derivatives[i] = np.dot(current_activations_reshaped, delta_reshaped)
             error = np.dot(delta, self.weights[i].T)
             
             if flag:
                 print("Derivatives for W{} : {}".format(i, self.derivatives[i]))             
             
         return error 
             
    def steepest_descent(self, learning_rate):
        for i in range(len(self.weights)):
            weights = self.weights[i]
#            print("Original W{} {}".format(i, weights))
            derivatives = self.derivatives[i]
            weights += derivatives * learning_rate
#            print("Updated W{} {}".format(i, weights))


    def _sigmoid(self, x):        
        return 1 / (1 + np.exp(-x))
    
    def _sigmoid_derivative(self, x):        
        return (x * (1.0 - x))
    
    def _mse(self, target, output):        
        return np.average((target - output)**2)
        
    
    def train(self, inputs, targets, epochs, learning_rate):
        
        for i in range(epochs):
            sum_error = 0
            for inp, target in (zip(inputs, targets)):
            
                output = self.forward_propogation(inp)
            
                error = target - output
            
                self.back_propagate(error)
            
                self.steepest_descent(learning_rate)
            
                sum_error += self._mse(target, output)
                
            av_error = sum_error/len(inputs)
            print("Average Error : {} at epoch {}".format(av_error, i))
                    


if __name__ == "__main__":
    

    training = load_wine()
    train_data = np.array(training['data'])
    labels = np.array(training['target'])
    
    x_train, x_test, y_train, y_test = train_test_split(train_data, labels, test_size=0.33)
        
    """
    Experiment 1 : Training the model without normalization, splitting 1/3rd of the dataset for training and the rest
    for testing, 13 inputs
    Uuncomment the below section and comment the other 3 sections
    """
    
    """    
    mlp = MLP()
    learning_rate=0.19  #hyper-parameter
    epochs=10000
    mlp.train(x_train, y_train, epochs, learning_rate)    
    
    # Test The Model
    
    output_test = []
#    print(input_test.shape)
#    t = np.array([2])
    for i in x_test:
        output = mlp.forward_propogation(i)
#        print(output)
        if output[0] > output[1] and output[0] > output[2]:
            output_test.append(0)
        elif output[1] > output[0] and output[1] > output[2]:
            output_test.append(1)
        elif output[2] > output[1] and output[2] > output[0]:
            output_test.append(2)
        else:
            output_test.append(2)

    print(confusion_matrix(y_test, output_test))
    
    """   

    
    """
    
    Experiment 2 :Training the model after linear normalization, splitting 1/3rd of the dataset for training and the rest
    for testing
    Uncomment the below section and comment the other 3 sections
    
    """
    
    """

    
    min_data = np.array([11.0,0.74,1.36,10.6,70.0,0.98,0.34,0.13,0.41,1.3,0.48,1.27,278])
    max_data = np.array([14.8,5.80,3.23,30.0,162.0,3.88,5.08,0.66,3.58,13.0,1.71,4.00,1680])
    
    data_r = np.array(max_data-min_data)           
    normalized_data = np.array((train_data-min_data)/data_r)

    xn_train, xn_test, yn_train, yn_test = train_test_split(normalized_data, labels, test_size=0.33)

    mlp = MLP()
    learning_rate=0.2  #hyper-parameter
    epochs=10000
    mlp.train(xn_train, yn_train, epochs, learning_rate)    
    
    outputn_test = []
    
    for i in xn_test:
        output = mlp.forward_propogation(i)
        if output[0] > output[1] and output[0] > output[2]:
            outputn_test.append(0)
        elif output[1] > output[0] and output[1] > output[2]:
            outputn_test.append(1)
        elif output[2] > output[1] and output[2] > output[0]:
            outputn_test.append(2)
        else:
            outputn_test.append(2)

    print(confusion_matrix(yn_test, outputn_test))



    """
    
    """
    
    Experiment 3 : Training the model after z-score normalization, splitting 1/3rd of the dataset for training and the rest
    for testing
    Uncomment the below section and comment the other 3 sections
    
    """    
    
    """
    mean_array = np.array([13.0,2.34,2.36,19.5,99.7,2.29,2.03,0.36,1.59,5.1,0.96,2.61,746])
    sd_array = np.array([0.8,1.12,0.27,3.3,14.3,0.63,1.00,0.12,0.57,2.3,0.23,0.71,315])
    
    znormalized_data = np.array((train_data-mean_array)/sd_array)
    
    xz_train, xz_test, yz_train, yz_test = train_test_split(znormalized_data, labels, test_size=0.33)
        
    mlp = MLP()
    learning_rate=0.22  #hyper-parameter
    epochs=10000
    mlp.train(xz_train, yz_train, epochs, learning_rate)    

    outputz_test = []
    
    for i in xz_test:
        output = mlp.forward_propogation(i)
        if output[0] > output[1] and output[0] > output[2]:
            outputz_test.append(0)
        elif output[1] > output[0] and output[1] > output[2]:
            outputz_test.append(1)
        elif output[2] > output[1] and output[2] > output[0]:
            outputz_test.append(2)
        else:
            outputz_test.append(2)

    print(confusion_matrix(yz_test, outputz_test))

        
    """
    
    """
    
    Experiment 4: Using Adam with categorical crossentropy loss function using Tensorflow and Keras
    Uuncomment the below section and comment the other 3 sections
    
    """
    
    np.random.seed(2)

    classifications = 3
    
    training = load_wine()
    train_data = np.array(training['data'])
    labels = np.array(training['target'])
    
    x_train, x_test, y_train, y_test = train_test_split(train_data, labels, test_size=0.4)
    
    # one hot encoding
    
    y_train = keras.utils.to_categorical(y_train-1, classifications)
    y_test = keras.utils.to_categorical(y_test-1, classifications)

    
    model = Sequential()
    model.add(Dense(13, input_dim = 13, activation='relu'))
    model.add(Dense(13, activation='relu'))
    model.add(Dense(classifications, activation='softmax'))
    
    
    tp = keras.metrics.TruePositives(thresholds=None, name=None, dtype=None)
    tn = keras.metrics.TrueNegatives(thresholds=None, name=None, dtype=None)
    fp = keras.metrics.FalsePositives(thresholds=None, name=None, dtype=None)
    fn = keras.metrics.FalseNegatives(thresholds=None, name=None, dtype=None)
    
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy',tp,tn,fp,fn])
    model.fit(x_train, y_train, batch_size=5, epochs=1300, validation_data=(x_test, y_test))
  