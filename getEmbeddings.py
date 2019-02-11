import re
import numpy as np
import os
from PIL import Image
import keras
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.layers import Activation
from keras.layers import Input, Lambda, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Reshape, GlobalAveragePooling2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras.models import load_model
import scipy.io as sio
#################################################### 
def read_file(filename):
    data=np.load(filename)
    return data

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 0.5
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)
################################### BASE NETWORK

def build_base_network():
    inputs=Input(shape=(40,200,1), name='in_layer') 
    #convolutional layer 1
    o1 = Conv2D(128, (5, 5), strides=(1, 1), padding="same", activation="relu")(inputs)
    o1 = Conv2D(128, (2, 1), strides=(2, 1), padding="same", activation='relu')(o1)
    o1 = Conv2D(128, (5, 5), strides=(1, 1), padding="same", activation="relu")(o1)
    o1 = Conv2D(128, (2, 1), strides=(2, 1), padding="same", activation='relu')(o1)
    o1 = Conv2D(128, (5, 5), strides=(1, 1), padding="same", activation="relu")(o1)
    o1 = Conv2D(128, (2, 1), strides=(2, 1), padding="same", activation='relu')(o1)
    o1 = Conv2D(128, (5, 5), strides=(1, 1), padding="same", activation="relu")(o1)
    o1 = Conv2D(128, (5, 1), strides=(5, 1), padding="same", activation='relu')(o1)
    o1 = Reshape((1,200,128))(o1)
    o1 = GlobalAveragePooling2D()(o1)
    o1=Dropout(0.25)(o1)
    o1 = Dense(256,activation='relu')(o1)
    o1=Dropout(0.25)(o1)
    o1 = Dense(128,activation='relu')(o1)
    model=Model(inputs,o1)
    return model

############################################################### Load network


model=build_base_network()
epochs = 30
rms = RMSprop()

model.compile(loss=contrastive_loss, optimizer=rms) ### NADAM optimizer
model.summary()
model.load_weights('base.h5')


################################################################################################# create train
train=[]
train_labels=[]

for i in range(40):
     path, dirs, files = next(os.walk("./new_test/s" + str(i+1)))
     file_count1 = len(files)
     for j in range(0,file_count1):
         img1 = read_file('new_test/s' + str(i+1) + '/' + str(j)+'.npy')
         train.append(img1)
         train_labels.append(i)


train=np.asarray(train)
train = np.expand_dims(train, axis=3)
train_labels=np.asarray(train_labels)
print(train.shape)  
     
             
train_embeddings=model.predict(train)

sio.savemat('./test_embedding.mat',{'test_embeddings':train_embeddings,'test_labels':train_labels})















