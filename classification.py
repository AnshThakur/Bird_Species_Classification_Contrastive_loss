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
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
################### load training embeddings

a=sio.loadmat('train_embedding.mat')

train=a['train_embeddings']
train_labels=a['train_labels']
train_labels=train_labels.transpose()
a=sio.loadmat('test_embedding.mat')

test=a['test_embeddings']
test_labels=a['test_labels']
test_labels=test_labels.transpose()
print(train.shape)
print(test.shape)


####################################### MLP classifier

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(512), random_state=1)
clf.fit(train,train_labels)

################################ prediction

pred=clf.predict(test)
print(classification_report(test_labels, pred))

















