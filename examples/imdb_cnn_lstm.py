'''Train a recurrent convolutional network on the IMDB sentiment
classification task.

Gets to 0.8498 test accuracy after 2 epochs. 41s/epoch on K520 GPU.
'''
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Convolution1D, MaxPooling1D
from keras.datasets import imdb
import tensorflow as tf
tf.python.control_flow_ops = tf

# Embedding
max_features = 20000
maxlen = 100
embedding_size = 128

# Convolution
filter_length = 3
nb_filter = 800
pool_length = 1

# LSTM
lstm_output_size = 200

# Training
batch_size = 20
nb_epoch = 500

'''
Note:
batch_size is highly sensitive.
Only 2 epochs are needed as the dataset is very small.
'''

print('Loading data...')
#(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features)
#print(len(X_train), 'train sequences')
#print(len(X_test), 'test sequences')
#print(X_train.shape, X_test.shape)
#print('Pad sequences (samples x time)')
#X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
#X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
#print('X_train shape:', X_train.shape)
#print('X_test shape:', X_test.shape)

X_train = np.load('/lfs/local/0/abhisg/vw/driver_data/full_driver_data_X_train')
X_train = X_train.reshape(X_train.shape[0],1,X_train.shape[1])
y_train = np.load('/lfs/local/0/abhisg/vw/driver_data/full_driver_data_Y_train')
X_test = np.load('/lfs/local/0/abhisg/vw/driver_data/full_driver_data_X_test')
X_test = X_test.reshape(X_test.shape[0],1,X_test.shape[1])
y_test = np.load('/lfs/local/0/abhisg/vw/driver_data/full_driver_data_Y_test')

print('Build model...')

model = Sequential()
#model.add(Embedding(max_features, embedding_size, input_length=maxlen))
model.add(Convolution1D(nb_filter,filter_length,input_shape=X_train.shape[1:],border_mode='same'))
#model.add(Convolution1D(nb_filter,filter_length,input_shape=(5,X_train.shape[-1]),border_mode='same'))
#model.add(Convolution1D(nb_filter=nb_filter,
#                        filter_length=filter_length,
#                        border_mode='valid',
#                        activation='relu',
#                        input_shape=X_train.shape[1:]
#                        ))
#model.add(MaxPooling1D(pool_length=pool_length))
model.add(LSTM(lstm_output_size))
model.add(Dropout(0.25))
model.add(Dense(2))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          validation_data=(X_test, y_test))
score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
val = model.predict(X_test)
counts = [0,0,0,0]
for i in xrange(len(val)):
    if y_test[i][1] == 1 and val[i][0] < val[i][1]:
        counts[0] += 1
    elif y_test[i][1] == 1 and val[i][0] >= val[i][1]:
        counts[1] += 1
    elif y_test[i][0] == 1 and val[i][0] > val[i][1]:
        counts[2] += 1
    elif y_test[i][0] == 1 and val[i][0] <= val[i][1]:
        counts[3] += 1
print(counts)
print('Test score:', score)
print('Test accuracy:', acc)
