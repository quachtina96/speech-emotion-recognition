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
	return random.randint(1,5) 

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

	train_baseline_labels = list_to_categorical(train_baseline_labels_list)
	train_glottal_labels = list_to_categorical(train_glottal_labels_list)
	train_spectrogram_labels = list_to_categorical(train_spectrogram_labels_list)
	test_baseline_labels = list_to_categorical(test_baseline_labels_list)
	test_glottal_labels = list_to_categorical(test_glottal_labels_list)
	test_spectrogram_labels = list_to_categorical(test_spectrogram_labels_list)

	train_dic = { 'train_baseline': train_baseline,
				  'train_glottal': train_glottal,
				  'train_spectrogram': train_spectrogram,
				  'train_baseline_labels': train_baseline_labels,
				  'train_glottal_labels': train_glottal_labels,
				  'train_spectrogram_labels': train_spectrogram_labels
				}
	test_dic = {  'test_baseline': test_baseline,
				  'test_glottal': test_glottal,
				  'test_spectrogram': test_spectrogram,
				  'test_baseline_labels': test_baseline_labels,
				  'test_glottal_labels': test_glottal_labels,
				  'test_spectrogram_labels': test_spectrogram_labels
				}
	return train_dic, test_dic

def load_datadic(i):
	filepath = '/home/akekeke/pickles/csv2pd/'
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

def calculateConfusionMatrix(predicted, actual):
	# predicted, actual are expected to be 2D, one hot arrays
	confusion = np.zeros((4,4))
	
	for i in range(n_test):
		label = predicted[i]
		idx = np.argmax(label)
		confusion[idx] += actual[i]

	count_sum = np.sum(predicted, axis = 0)
	pred_happy_count = count_sum[0]
	pred_sad_count = count_sum[1]
	pred_angry_count = count_sum[2]
	pred_neutral_count = count_sum[3]
	
	confusion[0] = confusion[0]/pred_happy_count
	confusion[1] = confusion[1]/pred_sad_count
	confusion[2] = confusion[2]/pred_angry_count
	confusion[3] = confusion[3]/pred_neutral_count

	return confusion

def calculateWeightedAccuracy():
	'''
	Weighted accuracy is the accuracy over all
	testing utterances in the dataset, and unweighted accuracy is the
	average accuracy over each emotion category (Happy, Angry,
	Sad and Neutral).
	'''
	return 0

def calculateUnweightedAccuracy():
	return 0