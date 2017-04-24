spectrogram = 0;
extract_covarep = 1;
calculate_mfcc = 0;

addpath(genpath('/Users/quacht/Documents/6.345/final_project/covarep/'));

data_wav_directory = '/Users/quacht/Documents/6.345/final_project/data/sentence_wav_sample/';
context_windows_dir = '/Users/quacht/Documents/6.345/final_project/data/spectrogram_context_windows/'
mfcc_dir = '/Users/quacht/Documents/6.345/final_project/data/mfcc_features/'
% glottal_source_dir = '/Users/quacht/Documents/6.345/final_project/data/glottal_source/'
fileID = fopen('/Users/quacht/Documents/6.345/final_project/data/EmoEvaluation/Categorical/test_utteranceIDs.txt','r');
% C is a cell array that holds a single entry--the contents of the file.
C = textscan(fileID,'%s');
% A list of every utteranceID
paths = C{1};


for i=1:length(paths)
    path = char(strcat(data_wav_directory, paths(i)));
    slash_occurences = strfind(path,'/');
    filename = path(slash_occurences(end)+1:length(path));
    
    [s, Fs] = audioread(path);
    seconds = length(s) / Fs;  % second
    milliseconds = seconds*1000;

    dcOffset = mean(s);
    s_offset = s - dcOffset;

    % In terms of samples...
    frame_length = 20/1000*Fs; %20ms
    frame_shift = 160; %10ms

    if spectrogram==1
        % Save spectrogram generated directly from the waveform to
        % .wav.spec.mat files
        context_windows = get_context_windows(s_offset, frame_length, frame_shift);
        save(strcat(context_windows_dir, filename, '.spec.mat'), 'context_windows')
    end
    
    if extract_covarep==1
        % Save mat files corresponding to each .wav file. .mat files contain the
%         feature matrix: features [number of frames X 35] and names:
%         containing the feature name correspond to each column of the
%         feature matrix
        sample_rate=0.01; % State feature sampling rate
        COVAREP_feature_extraction_on_file(path, sample_rate, frame_length, frame_shift);
    end
    
    if calculate_mfcc==1
        % TODO: Potentially switch this implementation of MFCC to the class
        % assignemt's implementation for calculating MFCC.
        [MFCC] = VAD_MFCC(s_offset,Fs)
        save(strcat(mfcc_dir, filename, '.mfcc.mat'), 'MFCC')
        disp('Finished MFCC')
    end
end








