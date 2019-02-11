import re
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.layers import Activation
from keras.layers import Input, Lambda, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Reshape, GlobalAveragePooling2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop

#################################################### ENERGY function
def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

########################## Contrastive Loss
def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 0.5
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)


######################################################

total_sample_size = 5000

############################################################# returns numpy array by reading image

def read_file(filename):
    data=np.load(filename)
    return data



#############################################################################

def get_data(total_sample_size):
    #read the mel specs
    mel = read_file('./new_train/s' + str(1) + '/' + str(0)+'.npy')
   
    #get the new size
    dim1 = mel.shape[0]
    dim2 = mel.shape[1]
    
    count = 0
    
    #initialize the numpy array with the shape of [total_sample, no_of_pairs, dim1, dim2]
    x_geuine_pair = np.zeros([total_sample_size, 2,1, dim1, dim2])  # 2 is for pairs
    y_genuine = np.zeros([total_sample_size, 1])
    
    for i in range(40):
        for j in range(int(total_sample_size/40)):
            ind1 = 0
            ind2 = 0
            
            #read images from same directory (genuine pair)
            path, dirs, files = next(os.walk("./new_train/s" + str(i+1)))
            file_count1 = len(files)
             

            while ind1 == ind2:
                ind1 = np.random.randint(4)
                ind2 = np.random.randint(4)
            
            # read the two images
        
            img1 = read_file('new_train/s' + str(i+1) + '/' + str(ind1 + 1)+'.npy')
            img2 = read_file('new_train/s' + str(i+1) + '/' + str(ind2 + 1)+'.npy')
            
            
            #store the images to the initialized numpy array
            x_geuine_pair[count, 0, 0, :, :] = img1
            x_geuine_pair[count, 1, 0, :, :] = img2
            
            #as we are drawing images from the same directory we assign label as 1. (genuine pair)
            y_genuine[count] = 1
            count += 1

    count = 0
    x_imposite_pair = np.zeros([total_sample_size, 2, 1, dim1, dim2])
    y_imposite = np.zeros([total_sample_size, 1])
    
    for i in range(int(total_sample_size/4)):
        for j in range(4):
            
            #read images from different directory (imposite pair)
            while True:
                ind1 = np.random.randint(40)
                ind2 = np.random.randint(40)
                if ind1 != ind2:
                    break
                    
            img1 = read_file('new_train/s' + str(ind1+1) + '/' + str(j + 1)+'.npy')
            img2 = read_file('new_train/s' + str(ind2+1) + '/' + str(j + 1)+'.npy')

            x_imposite_pair[count, 0, 0, :, :] = img1
            x_imposite_pair[count, 1, 0, :, :] = img2
            #as we are drawing images from the different directory we assign label as 0. (imposite pair)
            y_imposite[count] = 0
            count += 1
            
    #now, concatenate, genuine pairs and imposite pair to get the whole data
    X = np.concatenate([x_geuine_pair, x_imposite_pair], axis=0)
    Y = np.concatenate([y_genuine, y_imposite], axis=0)

    return X, Y


#############################################################

X, Y = get_data(total_sample_size)

print(X.shape)
print(Y.shape)

#############################
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.25)

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

############################################################### CREATE SIAMESE NET

img_a = Input(shape=(40,200,1))
img_b = Input(shape=(40,200,1))
base_network = build_base_network()
feat_vecs_a = base_network(img_a)
feat_vecs_b = base_network(img_b)
distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([feat_vecs_a, feat_vecs_b])
model = Model(input=[img_a, img_b], output=distance)

epochs = 20
rms = RMSprop()

model.compile(loss=contrastive_loss, optimizer=rms) ### NADAM optimizer
model.summary()




################################

# Create data for training

img_1 = x_train[:, 0]
img2 = x_train[:, 1]
img_1=np.rollaxis(img_1, 2, 1)
img_1=np.rollaxis(img_1, 3, 2)
print('final')
print(img_1.shape)
img2=np.rollaxis(img2, 2, 1)
img2=np.rollaxis(img2, 3, 2)
print('final')
print(img2.shape)
model.fit([img_1, img2], y_train, validation_split=.1,
          batch_size=64, verbose=1, nb_epoch=epochs)


base_network.save_weights('base.h5')


#########################################################
test=x_test[:, 0]
test=np.rollaxis(test, 2, 1)
test=np.rollaxis(test, 3, 2)

test1=x_test[:, 1]
test1=np.rollaxis(test1, 2, 1)
test1=np.rollaxis(test1, 3, 2)



pred = model.predict([test,test1])

###############################################
def compute_accuracy(predictions, labels):
    return labels[predictions.ravel() < 0.5].mean()

############################################

print(compute_accuracy(pred, y_test))

















