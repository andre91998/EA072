# -*- coding: utf-8 -*-

"""

for mean loss and accuracy, see initial.txt

"""

import tensorflow as tf
import os

AccuracySum=0
bestAccuracy=[1,0,0,0,0.2]

print("\nepoch: 5 ; neurons: 512 ; dropout: 0.5 \n")

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
model = tf.keras.models.Sequential([
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(512, activation=tf.nn.relu),
tf.keras.layers.Dropout(0.5),
tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
evaluation = model.evaluate(x_test, y_test)
model_json = model.to_json()

bestAccuracy = [1, 5, 512, 0.5, evaluation[0], evaluation[1]]

json_file = open("model_MLP.json", "w")
json_file.write(model_json)
json_file.close()
model.save_weights("model_MLP.h5")
print("Model saved to disk")
os.getcwd()

f=open("initial.txt","a+")
f.write(str(bestAccuracy[0])+" ; "+str(bestAccuracy[1])+" ; "+str(bestAccuracy[2])+" ; "+str(bestAccuracy[3])+" ; "+str(bestAccuracy[4])+" ; "+str(bestAccuracy[5])+"\n")
f.close()