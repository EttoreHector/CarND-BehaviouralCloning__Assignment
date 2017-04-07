# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 09:33:55 2017

@author: ettore
"""
#import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D


lines = []

# Read the file log containing references to the images and stearing angles
with open('./recovery/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# Split the images into training (80%) and test set (20%)
train_samples, validation_samples = train_test_split(lines, test_size=0.2)
 
# Correction for steering angle for left and right camera images
correction = 0.2


# This function provides the model with training examples in a way
# that does not consume too much memory resources.
def generator(lines, batch_size=32):
    num_train_samples = len(lines)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_train_samples, batch_size):
            batch_lines = lines[offset:offset+batch_size]
            
            # These lists will contain train samples and corresponding labels
            images = []
            angles = []
            
            for line in batch_lines:                
                # I read all three images, Centre (0), Left (1) and Right (2)
                for i in range(0,3):
                    # Read the steering angle and the image
                    angle = float(line[3])
                    image_path = './recovery/IMG/' + line[i].split('\\')[-1]
                    image = cv2.imread(image_path)
                    
                    # Flip the steering angle and the image
                    angle_flipped = -angle                                       
                    image_flipped = np.fliplr(image)
                    
                    # Append the original image and its mirrored version
                    # to the corresponding list
                    images.append(image)
                    images.append(image_flipped)
                    
                    # Adjust the steering angle depending whether the image
                    # was captured by the left or right camera
                    if i == 1:
                        angle += correction
                        angle_flipped -= correction
                    if i == 2:
                        angle -= correction
                        angle_flipped += correction
            
                    # Append the steering angle of the original and mirrored
                    # images to the corresponding list
                    angles.append(angle)
                    angles.append(angle_flipped)

            # Transform the lists of images and angles to numpy array
            # and feed them to the model.
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


# This function creates the model and load (if so specified) previously
# fitted parameters from a similar trained model
def create_model(weights_path=None):
    
    # This is the architecture of the model
    model = Sequential()
    model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: x/255.0 - 0.5))
    model.add(Convolution2D(24, 9, 9, subsample=(2, 2), border_mode='valid'))
    model.add(Convolution2D(36, 7, 7, subsample=(2, 2), border_mode='valid', 
                            activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid', 
                            activation='relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid'))
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Dropout(0.5))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(1))
    
    # Load the fitted parameters of a trained model with the same architecture
    if weights_path:
        model.load_weights(weights_path)
        
        # You can opt to freeze the weights of convolutional layers and
        # retrain only those of the fully connected layers.
        for i in range(len(model.layers)-6):
            model.layers[i].trainable = False

    return model
    

# Create, compile and train the model
model = create_model('./model.h5b')
model.compile(loss = 'mse', optimizer = 'adam')
history_object = model.fit_generator(train_generator, samples_per_epoch = 
                                     len(train_samples)*6, validation_data=
                                     validation_generator, nb_val_samples=
                                     len(validation_samples)*6, nb_epoch=4)

print(history_object.history.keys())

# S1ave the trained model
model.save('model.h5r')

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()