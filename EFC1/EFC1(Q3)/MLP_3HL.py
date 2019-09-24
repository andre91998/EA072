# -*- coding: utf-8 -*-
""""
%-----------------------------------------------------------------------------%
%Author: André Barros de Medeiros
%Date:09/14/2019
%Copyright: free to use, copy, and modify
%Description: Multi-Layer Perceptron to classify MNIST dataset images
%Important: Layers: 4 (512 neurons, 512, 10 neurons)
%           Epochs: 5
%           Dropout: 0.5 (first layer)
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

    
AccuracySum=0
bestAccuracy=[3,0,0,0,0.2]

for epoch in [7,8]:
    for neurons in [300,512]:
        for dropout in [0.2,0.3, 0.4, 0.5]:
        
            print("\nepoch: " + str(epoch)+"; neurons: "+str(neurons)+"; dropout: "+str(dropout) + "\n")
            
            (x_train, y_train),(x_test, y_test) = mnist.load_data()
            x_train, x_test = x_train / 255.0, x_test / 255.0
            
            model = tf.keras.models.Sequential([
             tf.keras.layers.Flatten(),
             tf.keras.layers.Dense(512, activation=tf.nn.relu),
             tf.keras.layers.Dropout(0.5),
             tf.keras.layers.Dense(neurons, activation=tf.nn.relu),
             tf.keras.layers.Dropout(dropout),
             tf.keras.layers.Dense(neurons, activation=tf.nn.relu),
             tf.keras.layers.Dropout(dropout),
             tf.keras.layers.Dense(10, activation=tf.nn.softmax)
            ])
            
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            model.fit(x_train, y_train, epochs=epoch)
            evaluation = model.evaluate(x_test, y_test) #store loss and accuracy
            
            print( "\nAcurracy with: " + str(epoch) + " , " + str(neurons) + " , " + str(dropout) + " is " + str(evaluation[1])+"\n")
            if evaluation[1] > bestAccuracy[4]:
                bestAccuracy = [2, epoch, neurons, dropout, evaluation[1]]
                print("\n New Best Accuracy \n")
            model_json = model.to_json()
            
            json_file = open("model_MLP.json", "w")
            json_file.write(model_json)
            json_file.close()
            
            model.save_weights("model_MLP.h5")
            print("Model saved to disk")
            os.getcwd()

f=open("three_hidden_layers.txt","w+")
f.write(str(bestAccuracy[0])+" ; "+str(bestAccuracy[1])+" ; "+str(bestAccuracy[2])+" ; "+str(bestAccuracy[3])+" ; "+str(bestAccuracy[4]));
f.close()            
#print(bestAccuracy)
