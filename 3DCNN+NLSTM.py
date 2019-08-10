
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
from keras.layers import Dense, Embedding,TimeDistributed,Conv3D,Input, Flatten,Reshape ,MaxPooling3D, Dropout,Activation, LSTM
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
time_v=15
pre_steps = 15
link_num = 278
test_sample_num = 6734
time_v=15

scaler = MinMaxScaler(feature_range=(0, 1))
trainX = np.load('./DATA/Train_data.npy')
trainX = trainX.reshape(sample_num*rows,cols)
trainX = scaler.fit_transform(trainX)
trainX = trainX.reshape(sample_num,rows,cols,1)
X_train=[]
for i in range(sample_num-timesteps-pre_steps+1):
   X_train.append(trainX[i:i+timesteps])
x_train = np.asarray(X_train,dtype='float32')
print ('x_train.shape',x_train.shape)
x_train=x_train.reshape((sample_num-timesteps-pre_steps+1,1,rows,cols,time_v,1))
print ('x_train.shape',x_train.shape)

y1 = np.load('./DATA/goal_train.npy')
y1 = np.asarray(y1).astype('float32')
y_train = []
for i in range(sample_num-timesteps-pre_steps+1):
    y_train.append(y1[i+timesteps:i+timesteps+pre_steps])
y_train = np.asarray(y_train).astype('float32')
y_train = y_train.reshape((sample_num -timesteps -pre_steps +1, link_num*pre_steps))
print ('y_train.shape',y_train.shape)

testX = np.load('./DATA/Test_data.npy')
testX = testX.reshape(test_sample_num*rows,cols)
testX = scaler.fit_transform(testX)
testX = testX.reshape((test_sample_num,rows,cols,1))
X_test=[]
for i in range(test_sample_num-timesteps-pre_steps+ 1):
   X_test.append(testX[i:i+timesteps])
x_test = np.asarray(X_test,dtype='float32')
x_test=x_test.reshape((test_sample_num-timesteps-pre_steps+1,1,rows,cols,time_v,1))
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
    
    
model = Sequential()
model.add(TimeDistributed(Conv3D(16, (3, 3, 3), padding='same'), input_shape=((1,rows,cols,time_v,1))))
model.add(Activation('relu'))
model.add(TimeDistributed(MaxPooling3D(pool_size=(2, 2, 2),padding='same')))


model.add(TimeDistributed(Conv3D(16, (3, 3, 3), padding='same')))
model.add(Activation('relu'))
model.add(TimeDistributed(MaxPooling3D(pool_size=(2, 2, 2),padding='same')))

model.add(TimeDistributed(Conv3D(16, (3, 3, 3), padding='same')))
model.add(Activation('relu'))
model.add(TimeDistributed(MaxPooling3D(pool_size=(2, 2, 2),padding='same')))

model.add(TimeDistributed(Conv3D(16, (3, 3, 3), padding='same')))
model.add(Activation('relu'))
model.add(TimeDistributed(MaxPooling3D(pool_size=(2, 2, 2),padding='same')))


model.add(TimeDistributed(Flatten()))
model.add(NestedLSTM(800, depth=2))
model.add(Dropout(0.2))
model.add(Dense(link_num*pre_steps))
model.add(Activation('linear'))
model.summary()
model = multi_gpu_model(model, gpus=2)
model.compile(optimizer=optimizers.RMSprop(lr=0.001),loss='mse',metrics=[mean_absolute_percentage_error])
model.fit(x_train, y_train,batch_size = 32,epochs=20,validation_data=(x_test, y_test),callbacks=[ModelCheckpoint('./3dcnn+nlstm_weights3.h5', monitor='val_loss',save_best_only=True, save_weights_only=True,verbose=1)])
model.load_weights('./3dcnn+nlstm_weights3.h5')
score = model.evaluate(x_test, y_test)
print('Test score:', score) 
 

