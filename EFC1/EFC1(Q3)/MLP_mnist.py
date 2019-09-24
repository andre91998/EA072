# -*- coding: utf-8 -*-
""""
%-----------------------------------------------------------------------------%
%Author: André Barros de Medeiros
%Date:09/23/2019
%Copyright: free to use, copy, and modify
%Description: Final Multi-Layer Perceptron to classify MNIST dataset images
%Important: Hidden Layers: 1 (300 neurons)
%           Epochs: 8
%           Dropout: 0.2 (first layer)
%           Activation Function: RELU (first layer)
%           Optimizer Algorithm: ADAM
%           Loss Function: Cross Entropy
%
% Averages (4 tests):
%           Loss:  / Accuracy: 
%-----------------------------------------------------------------------------%
"""

import tensorflow as tf
import os
mnist = tf.keras.datasets.mnist

bestAccuracy=[0,0,0,0,0]
            
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
 tf.keras.layers.Flatten(),
 tf.keras.layers.Dense(300, activation=tf.nn.relu), #hidden layer
 tf.keras.layers.Dropout(0.2), #hidden layer dropout
 tf.keras.layers.Dense(10, activation=tf.nn.softmax) #output layer
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=8)
evaluation = model.evaluate(x_test, y_test) #store loss and accuracy

print( "\nAcurracy with: 8 epochs, 300 neurons, and 0.2 dropout is " + str(evaluation[1])+"\n")
if evaluation[1] > bestAccuracy[4]:
    bestAccuracy = [1, 8, 300, 0.2, evaluation[0], evaluation[1]] #[hidden layers ; epochs ; neurons in hidden layer ; hidden layer dropout, loss, accuracy]
model_json = model.to_json()

json_file = open("model_MLP.json", "w")
json_file.write(model_json)
json_file.close()

model.save_weights("model_MLP.h5") #save weights to use 
print("Model saved to disk")
os.getcwd()

f=open("finalMLP.txt","a+")
f.write(str(bestAccuracy[0])+" ; "+str(bestAccuracy[1])+" ; "+str(bestAccuracy[2])+" ; "+str(bestAccuracy[3])+" ; "+str(bestAccuracy[4])+" ; "+str(bestAccuracy[5])+"\n");
f.close()            
#print(bestAccuracy)
