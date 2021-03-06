import pandas as pd
import pickle
import logging as log
import os
from pathlib import Path
from os.path import abspath
import sys

log.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
log.warning('is when this file started running.')

num = sys.argv[1]

dataPATH = '/pool001/quacht/IEMOCAP_full_release/Session'+num+'/data/'

print('loading labels from utterance_to_emotion_map.pickle')
# load a pickle
fd = pd.read_pickle(dataPATH+'utterance_to_emotion_map.pickle')
filename_to_label = []
data_filenames = []
labels = []
for key in fd:
    classifications = fd[key]
    count = [0]*4
    for emotions in classifications:
            if emotions[0] == 'Happiness':
                    count[0] += 1
            elif emotions[0] == 'Anger':
                    count[1] += 1
            elif emotions[0] == 'Sadness':
                    count[2] += 1
            elif emotions[0] == 'Neutral state':
                    count[3] += 1
    mode = max(count)
    if mode != 0:
            label_idx = count.index(mode)
            if label_idx == 0:
                    label = 0 # happiness
            elif label_idx == 1:
                    label = 1 # anger
            elif label_idx == 2:
                    label = 2 # sadness
            elif label_idx == 3:
                    label = 3 # neutral
            data_filenames.append(key)
            labels.append(label_idx)
            filename_to_label.append((key,label_idx))
                                                            
print('finished going thru label matrix ...')
print('length of filename_to_label list: ',len(filename_to_label))

baseline = pd.DataFrame()
glottal = pd.DataFrame()
spectrogram = pd.DataFrame()

final_labels = []

baseline_list = []
baseline_labels = []

glottal_list = []
glottal_labels = []

spectrogram_list = []
spectrogram_labels = []

log.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
log.warning('is when csv\'s started converting.')

count = -1
for name, label in filename_to_label:
        count = count + 1
        if count%200 == 0:
                print('count: ', count)
        # check if files exist
        file_baseline = dataPATH + 'baseline/' + name + '.wav.baseline.csv'
        file_glottal = dataPATH + 'glottal/' + name + '.wav.glott.csv'
        file_spectrogram = dataPATH + 'spectrogram/' + name + '.wav.spec.csv'
        if Path(file_baseline).exists() and Path(file_glottal).exists() and Path(file_spectrogram).exists():
                # if all files exists add it to dataset

                df_b = pd.read_csv(file_baseline, sep=',',header=None)
                #if df_b.shape[1] != 435*5:
                #       print(df_b.shape, ' not 435*5!!!')
                #baseline = baseline.concat(df, ignore_index=True)
                baseline_list.append(df_b.T)
                baseline_labels = baseline_labels + ([label]*df_b.shape[1])

                df_g = pd.read_csv(file_glottal, sep=',',header=None)
                #glottal = glottal.append(df, ignore_index=True)
                glottal_list.append(df_g.T)
                glottal_labels = glottal_labels + ([label]*df_g.shape[1])


                df_s = pd.read_csv(file_spectrogram, sep=',',header=None)
                #spectrogram = spectrogram.append(df, ignore_index=True)
                spectrogram_list.append(df_s.T)
                spectrogram_labels = spectrogram_labels + ([label]*df_s.shape[1])


                final_labels.append(label)
        else:
                print('one of the files didn\'t exist!')
                continue
baseline = pd.concat(baseline_list,ignore_index=True)
glottal = pd.concat(glottal_list,ignore_index=True)
spectrogram = pd.concat(spectrogram_list,ignore_index=True)

log.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
log.warning('is when this csv\'s finished converting.')



#print('baseline: ', baseline)

print('Verifying lengths ...')
print('baseline shape: ',baseline.shape)
print('baseline label shape: ', len(baseline_labels))

print('glottal shape: ', glottal.shape)
print('glottal_labels shape: ', len(glottal_labels))

print('spectrogram shape: ',spectrogram.shape)
print('spectrogram_labels: ',len(spectrogram_labels))

print('final_labels length: ',len(final_labels))


data_dictionary = { 'baseline': baseline, # pandas dataframe
                    'baseline_labels': baseline_labels,
                    'glottal': glottal, # pandas dataframe
                    'glottal_labels': glottal_labels,
                    'spectrogram': spectrogram, # pandas dataframe
                    'spectrogram_labels': spectrogram_labels,
                    'labels': final_labels # list for now?
                    }

print('pickling data_dictionary...')
with open('/home/akekeke/pickles/csv2pd/data_dic'+num+'.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(data_dictionary, f, pickle.HIGHEST_PROTOCOL)

log.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
log.warning('is when this file finished running.')
