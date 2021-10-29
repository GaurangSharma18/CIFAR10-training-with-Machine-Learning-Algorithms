
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 00:49:09 2021

@author: gaura
"""

import pickle
import numpy as np
from numpy.random import randint
from random import random
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import keras

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

def class_acc(pred,gt):
    errorSum=0;
    accuracyOfModal=0;
    for i in range(0,len(gt)):
        if gt[i]!=pred[i]:
            errorSum=errorSum+1;
    accuracyOfModal=(pred.shape[0]-errorSum)/(pred.shape[0])
    accuracyPercentage=accuracyOfModal*100
    return accuracyPercentage


a=[1,2,3,4,5];
X_all = np.array([], dtype = np.int32)
Y_all = np.array([], dtype = np.int32)
for i in a:
  datadict = unpickle('cifar-10-batches-py/data_batch_'+str(i));
  X = datadict["data"];
  Y = datadict["labels"];
  X_all=np.append(X_all,X);
  Y_all=np.append(Y_all,Y);

X_train =  np.array(X_all.reshape(50000, 3, 32, 32).transpose(0,2,3,1).astype("int32"));
Y_train =  np.array(Y_all);

datadict = unpickle('cifar-10-batches-py/test_batch');
X_test = datadict["data"];
Y_test = datadict["labels"];
X_test = X_test.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("int32");


#/******* define model ********/

model= tf.keras.models.Sequential();

# model.add(tf.keras.layers.Conv2D(32, (3,3), input_shape=(32,32,3), strides=(1,1), padding="valid", activation=tf.nn.relu));
# model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), padding="valid"));

# model.add(tf.keras.layers.Conv2D(32, (3,3), strides=(1,1), padding="valid", activation=tf.nn.relu));
# model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), padding="valid"));

# model.add(tf.keras.layers.Flatten());
# model.add(tf.keras.layers.Dense(32, activation=tf.nn.relu));
# model.add(tf.keras.layers.Dense(10, activation=tf.nn.sigmoid));


#import sys
#sys.exit();
model.add(tf.keras.layers.Flatten());
model.add(tf.keras.layers.Dense(64, input_dim=(32,32,3), activation=tf.nn.sigmoid));
model.add(tf.keras.layers.Dense(32, activation=tf.nn.sigmoid));
model.add(tf.keras.layers.Dense(16, activation=tf.nn.sigmoid));
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax));

#model.add(tf.keras.layers.Dense(128, input_dim=(32,32,3), activation=tf.nn.sigmoid))
#model.add(tf.keras.layers.Dense(64, input_dim=128, activation=tf.nn.sigmoid))
#model.add(tf.keras.layers.Conv2D(64, 5, strides=2, activation=tf.nn.relu))
#model.add(tf.keras.layers.Dense(32, input_dim=64, activation=tf.nn.sigmoid))
#model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
#/******* compile model ********/
#opt = tf.keras.optimizers.SGD(learning_rate=0.01)
opt = tf.keras.optimizers.SGD(learning_rate=0.001)
#model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])

model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

#model.compile(optimizer=opt, loss=tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error"), metrics=['accuracy'])

#/******* fit model ********/
history=model.fit(X_train,Y_train,epochs=40,batch_size=64,validation_split=0.2, shuffle=True);

# predictions=model.predict([X_test]);
# labeldict = unpickle('cifar-10-batches-py/batches.meta')
# label_names = labeldict["label_names"];
# print(label_names[np.argmax(predictions[0])])
# print("prediction is"+str(np.argmax(predictions[0])));


#plt.title(f"Image {i} label={label_names[Y_test[0]]} (num {Y[i]})")

print(model.summary());

model.save('CIFAR10SGDdense.model');
# list all data in history
predictions=model.predict([X_test]);
##print(np.argmax(predictions));
Y_predicted=predictions.argmax(axis=1);
#print(Y_predicted.shape)
testAccuracy=class_acc(Y_predicted,Y_test);
print("Test Accuracy is "+str(testAccuracy));

#print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'Validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'Validation'], loc='upper left')
plt.show()
#accuracy=class_acc(estimatedY,Y_test)
#print("Accuracy Of Task 1 is: "+str(accuracy))

#datadict = unpickle('/cifar-10-batches-py/test_batch')

#X = datadict["data"]
#Y = datadict["labels"]

#labeldict = unpickle('/content/drive/MyDrive/cifar-10-batches-py/batches.meta')
#label_names = labeldict["label_names"]
#print(X.shape)
#X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint32")
#X = X.reshape(10000, 3, 32, 32).transpose(0,1,2,3).astype("uint8")
#Y = np.array(Y)

#for i in range(X.shape[0]):
    # Show some images randomly
    #if random() > 0.999:
        #plt.figure(1);
        #plt.clf()
        #plt.imshow(X[i])
        #plt.title(f"Image {i} label={label_names[Y[i]]} (num {Y[i]})")
        #plt.pause(1)