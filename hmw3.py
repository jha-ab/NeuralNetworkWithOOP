#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 14:58:50 2020

@author: abhishek
"""

import tensorflow as tf
#import matplotlib 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#Load Mnist Dataset

mnist = tf.keras.datasets.mnist
(training_images, training_labels),(testing_images, testing_labels) = mnist.load_data()

#plt.imshow(training_images[0])
#print(training_labels[0])

# Set images to grayscale and normalize pixel values between 0 and 1

training_images = training_images.reshape(60000,28,28,1) #convert to greyscale
training_images = training_images/255.0 #Normalize
testing_images = testing_images.reshape(10000,28,28,1)
testing_images = testing_images/255.0

# Build the model

model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16,(3,3),activation='relu', input_shape=(28,28,1)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(16,(3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
        ])
model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

checkpoint_filepath = 'weights.{epoch:02d}.hdf5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='accuracy',
    mode='max',
    save_best_only=False)


model.fit(training_images, training_labels, batch_size=50, epochs=75, callbacks=[model_checkpoint_callback])
y_prob = model.predict(testing_images)
y_classes = y_prob.argmax(axis=-1)
print(classification_report(testing_labels, y_classes)) # generate classification report
c = confusion_matrix(testing_labels, y_classes) # Plotting the confusion matrix
print(c)
model.evaluate(testing_images, testing_labels, batch_size=50)