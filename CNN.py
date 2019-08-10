# -*- coding: utf-8 -*-

from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import numpy as np
from keras import models, optimizers
from keras import backend as K
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask   
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Lambda,TimeDistributed,Conv2D,Input, Flatten,Reshape ,MaxPooling2D, Dropout,Activation,LSTM,concatenate
from keras.callbacks import ModelCheckpoint
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


scaler = MinMaxScaler(feature_range=(0, 1))
rows = 164
cols = 148
sample_num = 14911
timesteps = 15
pre_steps = 1
#pre_steps is the prediction time.It can be 1 or 5 or 10
link_num = 278
test_sample_num = 6734 


trainX = np.load('./DATA/Train_data.npy')
trainX = trainX.reshape(sample_num*rows,cols)
trainX = scaler.fit_transform(trainX)
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
testX = testX.reshape(test_sample_num*rows,cols)
testX = scaler.fit_transform(testX)
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


cnn_input = Input(shape=(15,rows,cols,1))
CNN1 = TimeDistributed(Conv2D(16, (3, 3), padding='same'))(cnn_input)
Actvation1 = Activation('relu')(CNN1)
pooling1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2),padding='same'))(Actvation1)

CNN2 = TimeDistributed(Conv2D(32, (3, 3), padding='same'))(pooling1)
Actvation2 = Activation('relu')(CNN2)
pooling2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2),padding='same'))(Actvation2)

CNN3 = TimeDistributed(Conv2D(64, (3, 3), padding='same'))(pooling2)
Actvation3 = Activation('relu')(CNN3)
pooling3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2),padding='same'))(Actvation3)

CNN5 = TimeDistributed(Conv2D(128, (3, 3), padding='same'))(pooling3)
Actvation5 = Activation('relu')(CNN5)
pooling5 = TimeDistributed(MaxPooling2D(pool_size=(2, 2),padding='same'))(Actvation5)

x = Flatten()(pooling5)
x = Dropout(0.2)(x)
x = Dense(link_num*pre_steps)(x)
output = Activation("linear")(x) 

model = models.Model(inputs=cnn_input, outputs=output)
model.summary()
model = multi_gpu_model(model, gpus=2)

model.compile(optimizer=optimizers.RMSprop(lr=0.001),loss='mse',metrics=[mean_absolute_percentage_error])
print('Train...')
model.fit(x_train, y_train,batch_size = 32,epochs=20, validation_data=(x_test, y_test),callbacks=[ModelCheckpoint('./cnn_weights3.h5', monitor='val_loss',save_best_only=True, save_weights_only=True,verbose=1)])
model.load_weights('./cnn_weights3.h5')
print('evaluation.....')
score = model.evaluate(x_test, y_test)
print('Test score:', score)



