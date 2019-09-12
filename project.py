# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 15:56:32 2019

@author: User
"""

import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.utils import np_utils, print_summary
import pandas as pd
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import keras.backend as K

#reading the csv file
data = pd.read_csv('data.csv')
print(data.shape)

#making dataset out of the data 
dataset = np.array(data)

#randomly shuffle the dataset so that targets get mixed up
np.random.shuffle(dataset)
X = dataset
Y = dataset

#loading dataset's data and target into variables
X = X[:, 0:1024]
Y = Y[:, 1024]

#Splitting dataset into training and test set 
X_train = X[0:67000, :]
X_train = X_train / 255.

X_test = X[67000:72001, :]
X_test = X_test / 255.

Y = Y.reshape(Y.shape[0], 1)
Y_train = Y[0:67000, :]
Y_train = Y_train.T

Y_test = Y[67000:72001, :]
Y_test = Y_test.T

#printing the dimensions
print("number of training examples = " + str(X_train.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))

#x and y for shape of image
image_x = 32
image_y = 32

def keras_model(image_x,image_y):
    num_of_classes = 37
    model = Sequential()
    #adding pair of input layer and pooling layer into the model
    model.add(Conv2D(filters=32, kernel_size=(5, 5), input_shape=(image_x, image_y, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    #adding pair of convolutional and pooling layer into the model
    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))

    #flatten the model
    model.add(Flatten())
    model.add(Dense(num_of_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    #saving the model into a file
    filepath = "devanagari.h5"
    checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint1]
    
    return model, callbacks_list

#applying one hot encoding
train_y = np_utils.to_categorical(Y_train)
test_y = np_utils.to_categorical(Y_test)

#reshaping the train and test data and target 
train_y = train_y.reshape(train_y.shape[1], train_y.shape[2])
test_y = test_y.reshape(test_y.shape[1], test_y.shape[2])

X_train = X_train.reshape(X_train.shape[0], image_x, image_y, 1)
X_test = X_test.reshape(X_test.shape[0], image_x, image_y, 1)

print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(train_y.shape))

#feeding input into the keras model to extract model
model, callbacks_list = keras_model(image_x, image_y)

#fit the data into the extracted/formed model 
model.fit(X_train, train_y, validation_data=(X_test, test_y), epochs=1, batch_size=64,callbacks=callbacks_list)
scores = model.evaluate(X_test, test_y, verbose=0)

#printing error in data predicted
print("CNN Error: %.2f%%" % (100 - scores[1] * 100))
print_summary(model)
model.save('devanagari.h5')

    
    
