'''This example demonstrates the use of Convolution1D for text classification.

Gets to 0.89 test accuracy after 2 epochs.
90s/epoch on Intel i5 2.4Ghz CPU.
10s/epoch on Tesla K40 GPU.

'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding
from keras.layers import Convolution1D, MaxPooling1D
from keras.datasets import imdb
from keras import backend as K

import tensorflow as tf
tf.python.control_flow_ops = tf

# set parameters:
max_features = 5000
maxlen = 400
batch_size = 32
embedding_dims = 50
nb_filter = 1000
filter_length = 3
hidden_dims = 500
nb_epoch = 10

"""print('Loading data...')
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)"""

X_train = np.load('/lfs/local/0/abhisg/vw/driver_data/driver_data_X_train')
X_train = X_train.reshape(X_train.shape[0],1,X_train.shape[1])
y_train = np.load('/lfs/local/0/abhisg/vw/driver_data/driver_data_Y_train')
X_test = np.load('/lfs/local/0/abhisg/vw/driver_data/driver_data_X_test')
X_test = X_test.reshape(X_test.shape[0],1,X_test.shape[1])
y_test = np.load('/lfs/local/0/abhisg/vw/driver_data/driver_data_Y_test')

#X_train = X_train.reshape(-1, 1, X_train.shape[1])
#X_test = X_test.reshape(-1, 1, X_train.shape[1])
print('Build model...')
model = Sequential()
print(X_train.shape[1:])
model.add(Convolution1D(nb_filter,filter_length,input_shape=X_train.shape[1:],border_mode='same'))
print(model.output_shape)
#model.add(Dense(embedding_dims,input_dim=X_train.shape[1]))
# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
#model.add(Embedding(max_features,
#                    embedding_dims,
#                    input_length=maxlen,
#                    dropout=0.2))

# we add a Convolution1D, which will learn nb_filter
# word group filters of size filter_length:
#model.add(Convolution1D(nb_filter=nb_filter,
#                        filter_length=filter_length,
#                        border_mode='valid',
#                        activation='relu',
#                        subsample_length=1))
# we use max pooling:
model.add(MaxPooling1D(pool_length=1))
model.add(Convolution1D(nb_filter/2,filter_length,border_mode='same'))
model.add(MaxPooling1D(pool_length=1))

# We flatten the output of the conv layer,
# so that we can add a vanilla dense layer:
model.add(Flatten())

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.1))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(2))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(X_train, y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_data=(X_test, y_test))
