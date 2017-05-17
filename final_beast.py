from keras import backend as K

import pandas as pd
import numpy as np
import random
import h5py
import sys

import utils as utils

from pathlib import Path

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from keras import optimizers
from keras.models import Model
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Input
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from keras.utils import plot_model

import logging as log

import pydot
pydot.find_graphviz = lambda: True #this is a quick fix for keras visualization bug

#np.random.seed(12321)  # for reproducibility

modelpath = '/home/akekeke/pickles/models/'

log.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
log.warning('is when this file started running.')

f = sys.argv[2]

if sys.argv[1] == 'p':
	#if you want to poke/play with your model
	poke = True 
else:
	#if you dont want to poke with your model
	poke = False 

baseline = False
spectrogram = False
glottal = False

if f == 'b':
	baseline = True
	fe = 'baseline'
	print('Running baseline')
elif f == 's':
	spectrogram = True
	fe = 'spectrogram'
	print('Running spectrogram')
elif f == 'g':
	glottal = True
	fe = 'glottal'
	print('Running glottal')

# Load the dataset

train_dic, test_dic = utils.load_allData()
x_train = train_dic['train_'+fe]
x_test = test_dic['test_'+fe]
y_train = train_dic['train_'+fe+'_labels']
y_test = test_dic['test_'+fe+'_labels']

print('x_train.shape: ', x_train.shape)
print('x_test.shape: ' , x_test.shape)
print('y_train.shape: ', y_train.shape)
print('y_test.shape: ' , y_test.shape)


# and now onto the autoencoder

# for some reason, the code doesnt work with keras dataframes???
# converted into numpy arrays
X_train = X_train.values
X_test = X_test.values

# this is the size of our encoded representations
d = X_train.shape[1]
encoding_dim = 64
print('d: ',d)
print('encoding dim: ',encoding_dim)

if Path(modelpath + 'aec_'+fe+'.h5').exists() and !poke:
	print('loaded existing aec model')
	aec = load_model('aec_'+fe+'.h5')
else:
	#=================== Autoencoder ======================#
	if baseline:
		print('baseline autoencoder')
		# archticture is 435-160-64

		# this is our input placeholder
		input_f = Input(shape=(d,))
		h = Dense(160, activation='tanh')(input_f)
		# "encoded" is the encoded representation of the input	
		encoded = Dense(encoding_dim)(h)
		h = Dense(160, activation='tanh')(encoded)
		# "decoded" is the lossy lower dimension of the input
		decoded = Dense(d, activation='tanh')(h)

	if spectrogram or glottal:
		# archticture is 640-320-160-64

		# this is our input placeholder
		input_f = Input(shape=(d,)) #640
		h_320 = Dense(320, activation='tanh')(input_f) #320
		h_160 = Dense(160, activation='tanh')(h_320) #160
		# "encoded" is the encoded representation of the input
		encoded = Dense(encoding_dim)(h_160) #64
		h_160 = Dense(160, activation='tanh')(encoded) #160
		h_320 = Dense(320, activation='tanh')(h_160) #320
		# "decoded" is the lossy reconstruction of the input
		decoded = Dense(d, activation='tanh')(h_320)


	aec = Model(input_f, decoded)

	plot_model(aec, show_shapes = True, to_file='aec.png')

	print('autoencoder: ',aec.summary())
	aec.compile(loss='mean_squared_error', optimizer='sgd', metrics=["accuracy"])
	aec.fit(X_train, X_train,
	                epochs=5,
	                batch_size=256,
	                shuffle=True,
	                validation_data = (X_test,X_test))

	#Uncomment the following line if you would like to save your trained model
	aec.save(modelpath + './aec_'+fe+'.h5')


# this is to get encoded output
if baseline:
	aec2 = Model(input = aec.input, output = aec.layers[2].output)

if spectrogram or glottal:
	aec2 = Model(input = aec.input, output = aec.layers[3].output)


x_train_encoded = aec2.predict(x_train, batch_size = 256)
x_test_encoded = aec2.predict(x_test, batch_size = 256)

print('x_train_encoded.shape: ', x_train_encoded.shape)
print('x_test_encoded.shape: ', x_test_encoded.shape)

n_train = x_train_encoded.shape[0]
n_test = x_test_encoded.shape[0]
x_train_encoded = np.reshape(x_train_encoded, (n_train,1,encoding_dim))
x_test_encoded = np.reshape(x_test_encoded, (n_test,1,encoding_dim))

y_train_encoded = np.reshape(Y_train, (n_train,1,4))
y_test_encoded = np.reshape(Y_test, (n_test,1,4))

print('reshaped x_train_encoded.shape: ', x_train_encoded.shape)
print('reshaped x_test_encoded.shape: ', x_test_encoded.shape)
print('reshaped y_train_encoded.shape: ', y_train_encoded.shape)
print('reshaped y_test_encoded.shape: ', y_test_encoded.shape)


if Path(modelpath + 'model.h5_'+fe+'').exists() and !poke:
	print('loaded existing model')
	model = load_model('model_'+fe+'.h5')
else:
	#=================== Model ======================#
	# A setting of parameters used by us was the following:
	# The BLSTM-RNN has two layers, each of cell size 30, 
	# with a four-dimensional softmax layer on top for
	# emotional classification
	# To improve generalizability of the BLSTM, random noise is
	# added to the input sequences and the model weights in every
	# epoch, and can be controlled by the noise variance hyperparameters.

	# Parameters to test:	
	# • Learning Rate : [6e-6,8e-6,1e-5,2e-5,4e-5]
	# • Momentum : [0.7,0.8,0.9]
	# • Input noise variance : [0.0,0.1,0.2,0.3]
	# • Weight noise variance : [0.0,0.05,0.1,0.15,0.2]
	# • Batch size: 1300 utterances
	# • Maximum Epochs : 100
	batch_size = 1300

	model = Sequential()
	model.add( LSTM(units=30, input_shape=(None, encoding_dim), return_sequences=True) )
	model.add(Bidirectional(LSTM(units = 30, return_sequences=True)))
	model.add(Dense(4,activation = 'softmax',input_shape = (30,)))

	'''
	input_aec = Input(shape=(encoding_dim,))
	layer1 = Bidirectional(LSTM(units=30, input_shape=(batch_size, encoding_dim))) (input_aec)
	layer2 = Bidirectional(LSTM(units = 30)) (layer1)
	emotions = Dense(4,activation = 'softmax') (layer2)
	model = Model(input_aec, emotions)
	'''
	print('model summary: ', model.summary())
	plot_model(model, show_shapes = True, to_file='model.png')

	model.compile(loss='categorical_crossentropy', 
			optimizer=optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True), 
			metrics=["accuracy"])
	model.fit(x_train_encoded, y_train_encoded,
	      batch_size=1300,
	      epochs=75,
	      )

	model.save(modelpath + './model_'+fe+'.h5')

#evaulate
evaluate = model.evaluate(x_test_encoded, y_test_encoded, verbose=0)
print('eval: ',evaluate)


# evaluation


# confusion matrix 

# predicted	|				actual						
#			| happy		sad		angry 	neutral
# happy		|
# sad		|
# angry 	|
# neutral	|

confusion = np.zeros((4,4))

predicted = model.predict(x_test_encoded)
predicted = np.reshape(predicted,(n_test,4))
final_predicted = np.zeros((n_test,4))
final_predicted[np.arange(len(predicted)), predicted.argmax(1)] = 1
#final_predicted[np.where(predicted==np.max(predicted))] = 1

actual = y_test_encoded
actual = np.reshape(actual,(n_test,4))

y_true = np.argmax(actual, axis=1)
y_pred = np.argmax(final_predicted, axis=1)
emo_labels = ['Happy', 'Anger', 'Sad', 'Neutral']

print('final_predicted')
print(final_predicted)
print('actual')
print(actual)

print(np.sum(final_predicted, axis = 0))
cm = confusion_matrix(y_true, y_pred, labels = emo)
print(cm)

print(classification_report(y_true, y_pred, target_names=target_names))

log.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
log.warning('is when this file finished running.')