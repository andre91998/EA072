# -*- coding: utf-8 -*-
""""
%-----------------------------------------------------------------------------%
%Author: André Barros de Medeiros
%Date:09/16/2019
%Copyright: free to use, copy, and modify
%Description: Convolutional Network to classify MNIST dataset images
%Important: Layers: 3 (512 neurons, 512, 10 neurons)
%           Epochs: 5
%           Dropout: 0.5 (third layer)
%           Activation Function: RELU (first layer)
%           Optimizer Algorithm: ADAM
%           Loss Function: Cross Entropy
%
%           Loss: 0.0665 / Accuracy: 0.9811
%-----------------------------------------------------------------------------%
"""

import tensorflow as tf
import os
mnist = tf.keras.datasets.mnist

epoch=5
conv1=32
conv2=64
kernel= [3,3]
pool=[2,2]
dropout1=0.25
neurons=128
dropout2=0.5

bestAccuracy=[2,0,0,0,0,0,0,0,0.2]

 
print("\nepoch: " + str(epoch)+"\n Convolution 1: "+str(conv1)+" with kernel: "+str(kernel) +"\n Convolution 2: "+str(conv2)+" with kernel: "+str(kernel)+" and MaxPooling pool size: "+str(pool)+" and dropout: "+str(dropout1)+ "\n Fully connected layer with: "+str(neurons)+" and dropout: "+str(dropout2)+"\n")

(x_train, y_train),(x_test, y_test) = mnist.load_data()
# reshape to be [samples][width][height][pixels]
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
evaluation = model.evaluate(x_test, y_test)
print( "\nAcurracy is :" + str(evaluation[1])+"\n")
if evaluation[1] > bestAccuracy[4]:
    bestAccuracy = [2, epoch, conv1, conv2, kernel[1], pool[1], dropout1, neurons, dropout2, evaluation[0], evaluation[1]]
    print("\n New Best Accuracy \n")
model_json = model.to_json()

json_file = open("model_CNN.json", "w")
json_file.write(model_json)
json_file.close()
model.save_weights("model_CNN.h5")
print("Model saved to disk")
os.getcwd()

f=open("ConvInitial.txt","a+")
f.write("\n"+str(bestAccuracy[0])+" ; "+str(bestAccuracy[1])+" ; "+str(bestAccuracy[2])+" ; "+str(bestAccuracy[3])+" ; "+str(bestAccuracy[4])+" ; "+str(bestAccuracy[5])+" ; "+str(bestAccuracy[6])+" ; "+str(bestAccuracy[7])+" ; "+str(bestAccuracy[8])+" ; "+str(bestAccuracy[9])+" ; "+str(bestAccuracy[10]))
f.close()