from keras import backend as K

import pandas as pd
import numpy as np
import random
import h5py
import sys


from pathlib import Path

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder

from keras import optimizers
from keras.models import Model
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Input
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from keras.utils import plot_model

import pydot
pydot.find_graphviz = lambda: True #this is a quick fix for keras visualization bug

np.random.seed(12321)  # for reproducibility

f = sys.argv[1]

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

data_dic = pd.read_pickle('data_dic1.pickle')
print(type(data_dic))
print(data_dic.keys())
entire_x = data_dic['baseline']
entire_y = data_dic['baseline_labels']

# rand_idx = np.random.permutation(len(entire_x))
# entire_x = entire_x.iloc[rand_idx]
# entire_y = entire_y[rand_idx]

encoder = LabelEncoder()
encoder.fit(entire_y)
encoded_Y = encoder.transform(entire_y)
# convert integers to dummy variables (i.e. one hot encoded)
entire_y = np_utils.to_categorical(encoded_Y)
print('entire_y after encoding into one_hot: ',entire_y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(entire_x, entire_y, train_size = 0.9)
print('X_train.shape: ', X_train.shape)
print('X_test.shape: ', X_test.shape)
print('Y_train.shape: ', Y_train.shape)
print('Y_test.shape: ',Y_test.shape)


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

if Path('aec_'+fe+'.h5').exists():
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
	aec.save('./aec_'+fe+'.h5')


# this is to get encoded output
if baseline:
	aec2 = Model(input = aec.input, output = aec.layers[2].output)

if spectrogram or glottal:
	aec2 = Model(input = aec.input, output = aec.layers[3].output)


x_train_encoded = aec2.predict(X_train, batch_size = 256)
print(X_test.shape)
x_test_encoded = aec2.predict(X_test, batch_size = 256)

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


if Path('model.h5_'+fe+'').exists():
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

	#(#examples, #values in sequences, dim. of each value).
	#if baseline:
	#	(num_examples,1,435)
	#if spectrogram or glottal:
	#	(num_examples,1,640)

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

	print('x_traiaonsdf: ',x_train_encoded.shape)
	model.compile(loss='categorical_crossentropy', 
			optimizer=optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True), 
			metrics=["accuracy"])
	model.fit(x_train_encoded, y_train_encoded,
	      batch_size=1300,
	      epochs=20,
	      )

	model.save('./model_'+fe+'.h5')

#evaulate
evaluate = model.evaluate(x_test_encoded, y_test_encoded, verbose=0)
print('eval: ',evaluate)

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

'''
print('final_predicted')
print(final_predicted)
print('actual')
print(actual)

print(np.sum(final_predicted, axis = 0))

pred_happy_count = np.sum(final_predicted, axis = 0)[0]
pred_sad_count = np.sum(final_predicted, axis = 0)[1]
pred_angry_count = np.sum(final_predicted, axis = 0)[2]
pred_neutral_count = np.sum(final_predicted, axis = 0)[3]


for i in range(n_test):
	label = final_predicted[i]
	idx = np.argmax(label)
	confusion[idx] += actual[i]

	if idx == 0:	#happy
		confusion[0] += label
	elif idx == 1: #sad
		confusion[1] += label
	elif idx == 2: #angry
		confusion[2] += label
	elif idx == 3: #neutral
		confusion[3] += label

confusion[0] = confusion[0]/pred_happy_count
confusion[1] = confusion[1]/pred_sad_count
confusion[2] = confusion[2]/pred_angry_count
confusion[3] = confusion[3]/pred_neutral_count
'''



'''
df.iloc[np.random.permutation(len(df))]

nb_epoch = 3
batch_size = 64

feature_size = 87*5 
#=================== Autencoder ======================#
# A setting of parameters used by us was the following:
# Feedforward neural network with ONE hidden layer
# with activation  y_i = tanh(W*x_i+b)
# Output of autoencoder: z_i = W*y_i + b


#######################################################
from keras.layers import Input, Dense
from keras.models import Model

# this is the size of our encoded representations
encoding_dim = 

# this is our input placeholder
input_f = Input(shape=(784,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='tanh')(input_f)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(784)(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

model.save

x = ???
y = Dense(44, activation = 'tanh')(x)
z = Dense(435, activation = 'tanh')(y)

# SSE (Sum of Squared Error) was used for loss function
# tranined using backpropagation 
aec = Model(inputs = inputs, outputs = z)
aec.compile(loss='mean_squared_error', optimizer='sgd', metrics=["accuracy"])
model.fit(X_train, y_train, nb_epoch=nb_epoch, batch_size=batch_size,validation_split=0.1)

objective_score = model.evaluate(X_test, Y_test, batch_size=batch_size)

print('Evaluation on test set:', dict(zip(model.metrics_names, objective_score)))

# QUESTIONS TO ASK:
# assuming backpropagation = SGD?
# or use squared_hinge:  K.mean(K.square(K.maximum(1. - y_true * y_pred, 0.)), axis=-1)
# or use the one below?
def sum_squared_error(y_true, y_pred):
    return K.sum(K.square(y_pred - y_true), axis=-1)

#Uncomment the following line if you would like to save your trained model
#model.save('./current_model_conv.h5')
if K.backend()== 'tensorflow':
	K.clear_session
'''
