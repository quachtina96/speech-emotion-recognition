#!/bin/bash

# path_to_covarep=
# data_wav_directory=
# mat_dir=
# spectrogram_dir=
# glottal_dir=
# baseline_dir=
# path_to_utterance_IDs=

path_to_covarep='/Users/quacht/Documents/6.345/final_project/speech-emotion-recognition/covarep/'
data_wav_directory='/Users/quacht/Documents/6.345/final_project/data/sentence_wav_sample/'
mat_dir='/Users/quacht/Documents/6.345/final_project/data/mat/'
spectrogram_dir='/Users/quacht/Documents/6.345/final_project/data/spectrogram/'
glottal_dir='/Users/quacht/Documents/6.345/final_project/data/glottal/'
baseline_dir='/Users/quacht/Documents/6.345/final_project/data/baseline/'
path_to_utterance_IDs='/Users/quacht/Documents/6.345/final_project/data/EmoEvaluation/Categorical/test2_utteranceIDs.txt'

# /Applications/MATLAB_R2016b.app/bin/matlab -r -nodesktop 'try extract_experimental_data(\'$path_to_covarep\', \'$data_wav_directory\', "$mat_dirv", "$spectrogram_dir", "$glottal_dir", "$baseline_dir", "$path_to_utterance_IDs"); catch; end; quit'

/Applications/MATLAB_R2016b.app/bin/matlab -r -nodesktop extract_experimental_data(\'$path_to_covarep\', \'$data_wav_directory\', \'$mat_dirv\', \'$spectrogram_dir\', \'$glottal_dir\', \'$baseline_dir\', \'$path_to_utterance_IDs\')]
