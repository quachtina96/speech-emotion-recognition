#!/bin/bash

# extract.sh exists to submit a job for every list of utterances to analyze, 
# iterating through the different session numbers.


session_number=i
# path to the directory containing the lists of utterance IDs to be analyzed.
utterance_dir="/home/quacht/speech-emotion-recognition/session/session${session_number}/"

# for every list in the directory, pass that list to the slurm file.
cd $utterance_dir
for file in *; do
utterance_path=${utterance_dir}$file
echo $utterance_path
# sbatch /home/quacht/speech-remotion-recognition/extract_features.slurm $session_number $utterance_path
done

