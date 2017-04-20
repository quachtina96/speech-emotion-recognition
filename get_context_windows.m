function [context_windows] = get_context_windows(waveform, frame_length, frame_shift)
%get_context_windows Given a waveform, sampling frequency (Hz),
%frame_length (samples), frame_shift (samples), return a set
%of context_windows of size (640 x num_frames)
    s = waveform;
    % Only allow zero-padding in the last frame
    num_frames = ceil((length(s) - frame_length) / frame_shift) + 1;
    num_frames = max(num_frames, 1); % account for short waveforms

    % 0 pad so that the s_offset has 31200 samples instead of 31129
    last_sample_ind = 1 + (num_frames-1) *frame_shift+frame_length-1;
    padded_s_offset = cat(1, s_offset, zeros([last_sample_ind-num_frames, 1]));

    % Extract spectrogram from the speech waveform
    fft_size = 128*2;
    context_windows = [];
    current_context_window = [];
    for frame=1:num_frames
        frame_start = 1 + (frame-1) *frame_shift;
        frame_end = frame_start+frame_length-1;
        spectrum = abs(fft(padded_s_offset(frame_start:frame_end), fft_size));
        spectrum = spectrum(1:128);
        current_context_window = cat(1, current_context_window, spectrum);
        if mod(frame,5) == 0
            size(current_context_window)
            size(spectrum)
            context_windows = cat(2, context_windows, current_context_window);
            current_context_window = [];
        end
        plot(spectrum)
        if frame == num_frames
            % Zero-pad the last context frame so that it is size 640 x 1 
            pad = zeros([640-size_context(1),1]);
            context_windows = cat(2, context_windows, cat(1, current_context_window, pad));
            current_context_window = [];
        end    
    end

end

