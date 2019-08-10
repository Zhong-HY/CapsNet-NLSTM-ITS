# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 22:12:09 2019

@author: i
"""

from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import numpy as np
from keras import models, optimizers
from keras import backend as K
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask   
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding,TimeDistributed,Conv2D,Input, Flatten,Reshape ,MaxPooling2D, Dropout,Activation, LSTM
from keras.callbacks import ModelCheckpoint,LearningRateScheduler
from nested_lstm import NestedLSTM
import os
import argparse
from keras import callbacks
from keras.preprocessing.image import ImageDataGenerator
import time
import pandas as pd
import glob
from sklearn.preprocessing import MinMaxScaler
from keras.utils import multi_gpu_model

rows = 164
cols = 148
sample_num = 14911
timesteps = 15
pre_steps = 1
#pre_steps is the prediction time.It can be 1 or 5 or 10
link_num = 278
test_sample_num = 6734

trainX = np.load('./DATA/Train_data.npy')
trainX = trainX.reshape(sample_num,rows,cols,1)
X_train=[]
for i in range(sample_num-timesteps-pre_steps+1):
   X_train.append(trainX[i:i+timesteps])
x_train = np.asarray(X_train,dtype='float32')
print ('x_train.shape',x_train.shape)

y1 = np.load('./DATA/goal_train.npy')
y1 = np.asarray(y1).astype('float32')
y_train = []
for i in range(sample_num-timesteps-pre_steps+1):
    y_train.append(y1[i+timesteps:i+timesteps+pre_steps])
y_train = np.asarray(y_train).astype('float32')
y_train = y_train.reshape(sample_num -timesteps -pre_steps +1, link_num*pre_steps)
print ('y_train.shape',y_train.shape)


testX = np.load('./DATA/Test_data.npy')
testX = testX.reshape(test_sample_num,rows,cols,1)
X_test=[]
for i in range(test_sample_num-timesteps-pre_steps+ 1):
   X_test.append(testX[i:i+timesteps])
x_test = np.asarray(X_test,dtype='float32')
print ('x_test.shape',x_test.shape)

y_1 = np.load('./DATA/goal_test.npy')
y_1 = np.asarray(y_1).astype('float32')
y_test = []
for i in range(test_sample_num-timesteps-pre_steps+ 1):
    y_test.append(y_1[i+timesteps:i+timesteps+pre_steps])
y_test = np.asarray(y_test).astype('float32')
y_test = y_test.reshape(test_sample_num-timesteps-pre_steps+1, link_num*pre_steps)
print ('y_test.shape',y_test.shape)

def mean_absolute_percentage_error(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true), 1, np.inf))
    return 100. * K.mean(diff, axis=-1)

def CapsNet(input_shape, n_class, routings):
    
    x = Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 =Conv2D(filters=128, kernel_size=9,strides=2, padding='valid', activation='relu', name='conv1')(x)

   
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=16, kernel_size=9, strides=4, padding='valid')

    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
                             name='digitcaps')(primarycaps)
   
    
    out_caps = Flatten()(digitcaps)
    # outputs = Dense(278)(out_caps)
   
      
    # Models for training and evaluation (prediction)
    train_model = models.Model(x,  out_caps)
    
    return train_model

print('Build model...')
capsnet = CapsNet(input_shape=[rows,cols,1],n_class=30,routings=3)
capsnet.summary()
model = Sequential()
model.add(TimeDistributed(capsnet,input_shape = (timesteps,rows,cols,1)))
model.add(LSTM(800,return_sequences=True))
model.add(LSTM(800))
model.add(Dropout(0.2))
model.add(Dense(link_num*pre_steps))
model.summary()
model = multi_gpu_model(model, gpus=4)
print('Train...')
model.compile(optimizer=optimizers.RMSprop(lr=0.001),loss='mse',metrics = [mean_absolute_percentage_error])

print('Train...')
model.fit(x_train, y_train,
          batch_size =32,
          epochs=20, validation_data=(x_test, y_test),
          callbacks=[ModelCheckpoint('./capsnet+lstm_weights3.h5', monitor='val_loss' , save_best_only=True, save_weights_only=True,verbose=1)])

model.load_weights('./capsnet+lstm_weights3.h5')
print("Test...")
score = model.evaluate(x_test, y_test)
print('Test score:', score)
