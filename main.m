spectrogram = 0;
glottal_source = 1;

addpath(genpath('/Users/quacht/Documents/6.345/final_project/covarep/'));

data_wav_directory = '/Users/quacht/Documents/6.345/final_project/data/sentence_wav_sample/';
context_windows_dir = '/Users/quacht/Documents/6.345/final_project/data/spectrogram_context_windows/'
glottal_source_dir = '/Users/quacht/Documents/6.345/final_project/data/glottal_source/'
fileID = fopen('/Users/quacht/Documents/6.345/final_project/data/EmoEvaluation/Categorical/test_utteranceIDs.txt','r');
% C is a cell array that holds a single entry--the contents of the file.
C = textscan(fileID,'%s');
% A list of every utteranceID
paths = C{1};


for i=1:length(paths)
    path = char(strcat(data_wav_directory, paths(i)));
    slash_occurences = strfind(path,'/');
    filename = path(slash_occurences(end):length(path))
    
    [s, Fs] = audioread(path);
    seconds = length(s) / Fs;  % seconds
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
    % Extract from the glottal source waveform
    if glottal_source==1
        [g,dg,a,ag] = iaif(s_offset,Fs,p_vt,p_gl,d,hpfilt);
        context_windows = get_context_windows(g, frame_length, frame_shift);
        save(strcat(context_windows_dir, filename, '.glott.mat'), 'context_windows')
    end
    
end




