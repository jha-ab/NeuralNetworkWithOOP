#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 18:27:28 2020

@author: abhishek
"""

import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
np.random.seed(2)
classifications = 10

train_images = train_images.reshape(60000,28,28,1) #convert to greyscale
train_images = train_images/255.0 #Normalize
test_images = test_images.reshape(10000,28,28,1)
test_images = test_images/255.0


tr_cat = tf.keras.utils.to_categorical(train_labels, classifications)
ts_cat = tf.keras.utils.to_categorical(test_labels, classifications)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
  tf.keras.layers.Dense(1024,activation='relu'),
  tf.keras.layers.Dense(128,activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.summary()
model.compile(optimizer='Nadam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

checkpoint_filepath = 'weights.{epoch:02d}.hdf5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='accuracy',
    mode='max',
    save_best_only=False)


model.fit(train_images, train_labels, batch_size=50, epochs=90, callbacks=[model_checkpoint_callback])
y_prob = model.predict(test_images)
y_classes = y_prob.argmax(axis=-1)
print(classification_report(test_labels, y_classes)) # generate classification report
c = confusion_matrix(test_labels, y_classes) # Plotting the confusion matrix
print(c)
model.evaluate(test_images, test_labels, batch_size=50)