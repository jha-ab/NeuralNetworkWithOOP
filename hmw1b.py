#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 20:51:04 2020

@author: Abhishek Jha

HMW1 - Problem 2

Code the backpropagation algorithm and test it in the following 2 class problem 
called the Star problem. Use a single hidden layer MLP and specify the size of the 
hidden layer and tell why you select that number of hidden PEs.

x1   x2      d
1     0        1
0     1        1
-1    0        1
0    -1        1
0.5     0
-.5   0.5     0
-.5     0
-.5   -.5     0

I expect that the system learns this pattern exactly. You can think that points close 
to the x/y axes belong to one class and all the others belong to another class (hence the star). 
See how well your solution performs in points that do not belong to the training set, 
by selecting other points that follow the pattern (keep the points within the square 
of size 2, center at the origin). How could you improve the generalization accuracy of 
the solution? Show experimentally how your suggestion works. 


""" 

import numpy as np

class MLP:
    
    def __init__(self, ip=2, hidden=[8], op=1):
        
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
    
    
    #Create a MLP and train the model
    # Use this section to modify the Learning rate
    mlp = MLP()
    learning_rate=0.15 #hyper-parameter
    epochs=10000
    inputs = np.array([[1,0],[0,1],[-1,0],[0,-1],[5,0.5],[-0.5,0.5],[5,-0.5],[-0.5,-0.5]])
    d = np.array([[1],[1],[1],[1],[0],[0],[0],[0]])
    mlp.train(inputs, d, epochs, learning_rate)    
    
    # Test The Model
    
    input_test = np.array([0,1])
    t = np.array([1])
    output = mlp.forward_propogation(input_test)    
    print("Input, Expected, Actual")
    print(input_test, t, output)
    
    
    