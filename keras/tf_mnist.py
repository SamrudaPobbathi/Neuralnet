import tensorflow as tf
import keras
import numpy as np
import statistics
import h5py

from os import path
from time import time
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.callbacks import EarlyStopping

#Load dataset (choose one)
(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()
#(x_train, y_train),(x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

#Data loading and preprocessing
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#Create the model 
model = tf.keras.models.Sequential([
  tf.keras.layers.InputLayer(input_shape=(784,)),
  tf.keras.layers.Dense(units=300, activation='tanh'),
  tf.keras.layers.Dense(units=10, activation='softmax'),
])

#Compile the model
model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Fit with Tensorboard
#model.fit(x_train, y_train, epochs=50,  batch_size=112, callbacks=[tf.keras.callbacks.TensorBoard(log_dir="logs/final/{}".format(time()))])

#Fit and get the data
numtest = 1
numepochs = 1
for epoch in range(numtest):
  res_train = model.fit(x_train, y_train, epochs=numepochs,  batch_size=112, callbacks=[EarlyStopping(monitor='accuracy', patience=1)])
  res_test = model.evaluate(x_test, y_test)
  print("The test accuracy is ",res_test[1],"\n")

#Save   
model.save_weights("modelweights.h5")  
with open('model_architecture.json', 'w') as f:
    f.write(model.to_json())