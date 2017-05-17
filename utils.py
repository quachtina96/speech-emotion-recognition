import pandas as pd
import numpy as np
import random
import h5py

from pathlib import Path

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder

from keras.utils import np_utils
from keras.models import load_model
from keras.utils import plot_model

def pickTestSession():
	return random.randomint(1,5) 

def load_allData():
	testSesh = pickTestSession()
	print('Session '+str(testSesh)+' used as test set')
	for i in range(1,6):
		data_dic = load_datadic(i)

		baseline = data_dic['baseline']
		glottal = data_dic['glottal']
		spectrogram = data_dic['spectrogram']
		baseline_labels = data_dic['baseline_labels']
		glottal_labels = data_dic['glottal_labels']
		spectrogram_labels = data_dic['spectrogram_lables']

		if i == testSesh:
			test_baseline_list.append(baseline)
			test_baseline_labels_list.append(baseline_labels)
			test_glottal.append(glottal)
			test_glottal_labels_list.append(glottal_labels)
			test_spectrogram.append(spectrogram) 			
			test_spectrogram_labels_list.append(spectrogram_labels) 
		else:
			train_baseline_list.append(baseline)
			train_baseline_labels_list.append(baseline_labels)
			train_glottal.append(glottal)
			train_glottal_labels_list.append(glottal_labels)
			train_spectrogram.append(spectrogram) 			
			train_spectrogram_labels_list.append(spectrogram_labels) 

	train_baseline = pd.concat(train_baseline_list,ignore_index=True)
	train_glottal = pd.concat(train_glottal_list,ignore_index=True)
	train_spectrogram = pd.concat(train_spectrogram_list,ignore_index=True)
	test_baseline = pd.concat(test_baseline_list,ignore_index=True)
	test_glottal = pd.concat(test_glottal_list,ignore_index=True)
	test_spectrogram = pd.concat(test_spectrogram_list,ignore_index=True)

	train_dic = { 'train_baseline': 0,
				  'train_glottal': 0,
				  'train_spectrogram': 0,
				  'train_labels': 0,
				}
	test_dic = { 'test_baseline': 0,
				  'test_glottal': 0,
				  'test_spectrogram': 0,
				  'test_labels': 0,
				}
	return train_dic, test_dic

def load_datadic(i):
	data_dic = pd.read_pickle('data_dic'+str(i)+'.pickle')
	return data_dic

def list_to_categorical(y):
	print('input y shape: ', y.shape)
	encoder = LabelEncoder()
	encoder.fit(y)
	encoded_Y = encoder.transform(y)
	# convert integers to dummy variables (i.e. one hot encoded)
	hot_y = np_utils.to_categorical(encoded_Y)
	print('y.shape after encoding into one_hot: ',hot_y.shape)
	return hot_y

def reshape(x):
	n,d = x.shape
	x_reshaped = np.reshape(x, (n,1,d))
