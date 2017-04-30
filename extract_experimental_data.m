function [ success ] = extract_experimental_data(path_to_covarep, data_wav_directory, mat_dir, spectrogram_dir, glottal_dir, baseline_dir, path_to_utterance_IDs)
%extract_experimental_data given a list of utterance_IDs and the 
%   Detailed explanation goes here
success = 0;
addpath(genpath(path_to_covarep));

fileID = fopen(path_to_utterance_IDs);
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


    % Save spectrogram generated directly from the waveform to
    % .wav.spec.mat files
    spec_context_windows = get_context_windows(s_offset, frame_length, frame_shift);
   
    [MFCC] = VAD_MFCC(s_offset,Fs);

    % Save mat files corresponding to each .wav file. .mat files contain the
%         feature matrix: features [number of frames X 35] and names:
%         containing the feature name correspond to each column of the
%         feature matrix
    sample_rate=0.01; % State feature sampling rate to be 10 ms
    mfcc_size = size(MFCC);
    num_frames = mfcc_size(2);
    [glottal_context_windows, voice_feature_names, voice_features] = COVAREP_feature_extraction_on_file(path, sample_rate, frame_length, frame_shift, num_frames);
    voice_features = transpose(voice_features);
    % Concatenate the MFCCs and the Voice Quality features to create
    % the baseline features
    baseline_feats = [MFCC; voice_features];

    % Create context windows using the these features.
    baseline_context_windows = [];
    current_context_window = [];
    for frame=1:num_frames
        current_context_window = cat(1, current_context_window, baseline_feats(:,frame));
        if mod(frame,5) == 0
            baseline_context_windows = cat(2, baseline_context_windows, current_context_window);
            current_context_window = [];
        end
        if frame == num_frames
            % Zero-pad the last context frame so that it is size 640 x 1
            size_context = size(current_context_window);
            % 435 should be the size of the context frames for baseline
            % features because MFCC's have a length of 13 and the count of the
            % rest of the features is 74. 
            pad = zeros([435-size_context(1),1]);
            baseline_context_windows = cat(2, baseline_context_windows, cat(1, current_context_window, pad));
            current_context_window = [];
        end    
    end

    % Save context window and glottal waveform to a file.
    mfcc_names = {'MFCC_1','MFCC_2','MFCC_3','MFCC_4','MFCC_5', ...
    'MFCC_6','MFCC_7','MFCC_8','MFCC_9','MFCC_10','MFCC_11','MFCC_12', ...
    'MFCC_13'};
    names_to_save = [mfcc_names, voice_feature_names];

    % Write feature names to text file
    fileID = fopen('baseline_features_names.txt','w');
    formatSpec = '%s,';
    [nrows,ncols] = size(names_to_save);
    for col = 1:ncols
        fprintf(fileID,'%s\n',names_to_save{col});
    end
    fclose(fileID);

    % Save the data to .mat file specific to this wav file.
    save(strcat(mat_dir, filename, '.mat'),'baseline_context_windows','names_to_save','glottal_context_windows', 'spec_context_windows');

    % Write the glottal and baseline context windows each to a csv file.
    csvwrite(strcat(baseline_dir, filename, '.baseline.csv'),baseline_context_windows);
    csvwrite(strcat(glottal_dir, filename, '.glott.csv'),glottal_context_windows);
    csvwrite(strcat(spectrogram_dir, filename, '.spec.csv'),spec_context_windows);
    success = 1;
end
end

